import torch
import torch.nn as nn
import torch.nn.functional as F

# BasicExpert
# Minimal expert implementation: single linear transformation.
# This represents one expert in the MoE layer.
class BasicExpert(nn.Module):
    def __init__(self,feature_in,feature_out):
        super().__init__()
        # Linear projection: (batch, in_features) -> (batch, out_features)
        self.fc=nn.Linear(feature_in,feature_out)
        
    def forward(self,x):
        return self.fc(x)
       
class BasicMoE(nn.Module):
    def __init__(self,feature_in,feature_out,num_experts):
        super().__init__()
        self.gate=nn.Linear(feature_in,num_experts)
        self.experts=nn.ModuleList(
            BasicExpert(
                feature_in,feature_out
            ) for _ in range(num_experts)
            
        )
    def forward(self,x):
        logits=self.gate(x)
        #dense MoE
        expert_output_list=[
            expert(x)for expert in self.experts
            
        ]
        expanded_expert_outputs=[
            expert_output.unsqueeze(1)
            for expert_output in expert_output_list
        ]
        stacked_expert_outputs=torch.concat(
            expanded_expert_outputs,dim=1
        )
        
        expert_weights=F.softmax(logits,dim=1)
        expert_weights=expert_weights.unsqueeze(1)
        output=expert_weights@stacked_expert_outputs
        return output.squeeze(1) # (batch, out_features)
    
def test_basic_moe():
    x=torch.rand(4,512)
    basic_moe=BasicMoE(512,128,4)
    output=basic_moe(x)
    print(output.shape)
    
    
test_basic_moe()
    
    