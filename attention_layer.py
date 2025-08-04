import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size=1
seq_len=10 # for now 
d_model = 512  
no_of_head=8
d_k = d_model // no_of_head


class calculate_attention_scores(nn.Module): #use this class for all pytorch functionalities for neural networks
    def __init__(self,d_model, no_of_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.no_of_head = no_of_head
        self.d_k = d_model // no_of_head
        self.w_q=nn.Linear(self.d_model,self.d_model,bias=False)  ## create a learnable paramerter for q with random weights initialized
        self.w_k=nn.Linear(self.d_model,self.d_model,bias=False)
        self.w_v=nn.Linear(self.d_model,self.d_model,bias=False)
        self.fc = nn.Linear(d_model, d_model)


    def forward(self,x):

        #x has shape (batch_size, seq_len, d_model) so
        batch_size,seq_len,_=x.size() #note we use _ for embedding dim as we don't use it here


        #do linear projection
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)

        #reshape it like we have (batch_size,seq_len,d_model) this is what we get fromt the embedding layer
        #now what we do is that we split the d_model into no_of_head*d_k then
        #we transpose it to (batch_size,no_of_head,seq_len,d_k) 
        q=q.view(batch_size,seq_len,no_of_head,d_k).transpose(1,2)
        k=k.view(batch_size,seq_len,no_of_head,d_k).transpose(1,2)
        v=v.view(batch_size,seq_len,no_of_head,d_k).transpose(1,2)

        #lets calculate the attention score
        attention_score=torch.matmul(q,k.transpose(-2,-1)/(self.d_k**0.5)) 
        atn=F.softmax(attention_score,dim=-1)
        context = torch.matmul(attn, v)  
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.fc(context)
        return output, attn