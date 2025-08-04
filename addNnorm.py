import torch.nn as nn

class AddAndNorm(nn.Module):
    def __init__(self, d_model):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, sublayer_output):
        added = x + sublayer_output   # Residual connection
        normalized = self.norm(added) # Layer normalization
        return normalized
