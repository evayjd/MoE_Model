from basic_modle import BasicExpert
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEConfig:
    def __init__(self, hidden_dim, expert_number, top_k, shared_experts_number=2):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number


class MoERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.expert_number)
        self.expert_number = config.expert_number
        self.top_k = config.top_k

    def forward(self, x):
        """
        x: (N, hidden_dim)  where N = batch_size * seq_len
        """
        router_logits = self.gate(x)                     # (N, expert_number)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, expert_number)

        # top-k experts per token
        router_weights, selected_experts_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # (N, top_k), (N, top_k)

        # normalize top-k weights so they sum to 1 per token
        router_weights = router_weights / torch.sum(router_weights, dim=-1, keepdim=True)
        router_weights = router_weights.to(x.dtype)      # keep dtype consistent

        # one-hot mask: (N, top_k, expert_number)
        expert_mask = F.one_hot(selected_experts_indices, num_classes=self.expert_number)

        # permute to: (expert_number, top_k, N)
        expert_mask = expert_mask.permute(2, 1, 0)

        return router_logits, router_weights, selected_experts_indices, expert_mask


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number

        self.experts = nn.ModuleList(
            BasicExpert(config.hidden_dim, config.hidden_dim)
            for _ in range(config.expert_number)
        )

        self.router = MoERouter(config)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape  
        hidden_states = x.view(-1, hidden_dim)      # (N, hidden_dim), N=batch*seq


        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)


        final_hidden_states = torch.zeros(
            batch_size * seq_len, self.hidden_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )  # (N, hidden_dim)

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            current_expert_mask = expert_mask[expert_idx]  # (top_k, N), 0/1

            # positions where this expert is selected:
            # top_idx: which slot in top_k (0..top_k-1)
            # token_idx: which token index in [0..N-1]
            top_idx, token_idx = torch.where(current_expert_mask)

            if token_idx.numel() == 0:
                continue

            
            current_state = hidden_states[token_idx]       # (M, hidden_dim)
            current_state = expert_layer(current_state)    # (M, hidden_dim)

            # router weight for each (token, slot)
            current_token_router_weight = router_weights[token_idx, top_idx]  # (M,)
            current_token_router_weight = current_token_router_weight.unsqueeze(-1)  # (M, 1)

            current_hidden_states = current_state * current_token_router_weight  # (M, hidden_dim)

            # accumulate into final output
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(hidden_states.dtype)
            )


        final_hidden_states = final_hidden_states.view(batch_size, seq_len, self.hidden_dim)
        return final_hidden_states, router_logits, router_weights, selected_experts_indices, expert_mask


def test_sparse_moe():
    config = MoEConfig(hidden_dim=512, expert_number=4, top_k=2)
    sparse_moe = SparseMoE(config)
    x = torch.rand(4, 16, 512)
    output = sparse_moe(x)
    print(output[0].shape)

test_sparse_moe()
