import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)

    def create_padding_mask(self, input_ids, pad_token_id=0):
        padding_mask = (input_ids == pad_token_id)  # Here we notify where and all we have padding tokens so true if padding token else false 
        return padding_mask.unsqueeze(1).unsqueeze(2)


    def create_lookahead_mask(self, seq_len, device):
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool() #creates a traignluar ones mask
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, input_ids):
        batch_size, seq_len, _ = x.size()
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        padding_mask = self.create_padding_mask(input_ids, pad_token_id=0)
        lookahead_mask = self.create_lookahead_mask(seq_len, x.device)
        combined_mask = padding_mask | lookahead_mask

        scores = scores.masked_fill(combined_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.fc(context)
        return output, attn
