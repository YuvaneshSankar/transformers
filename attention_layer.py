import torch
import torch.nn as nn


batch_size=1
seq_len=10 # for now 
d_model = 512  
no_of_head=8
d_k = d_model // no_of_head


class calculate_attention_scores(seq_len, d_model,no_of_head,x):
    def __init__(self, seq_len, d_model, no_of_head):
        self.seq_len = seq_len
        self.d_model = d_model
        self.no_of_head = no_of_head
        self.d_k = d_model // no_of_head
        self.w_q=nn.Linear(self.d_model,self.d_model,bias=False)  ## create a learnable paramerter for q with random weights initialized
        self.w_k=nn.Linear(self.d_model,self.d_model,bias=False)
        self.w_v=nn.Linear(self.d_model,self.d_model,bias=False)

    def attention_matrics(self,self.w_q,self.w_k,self.w_v,x):
        self.q=self.w_q(x)
    def calculate_attention_scores(self,self.q,self.k,self.d_model):
        self.scores=(q* k.transpose(-2,-1)) / (d_model ** 0.5)  