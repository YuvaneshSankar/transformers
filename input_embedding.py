# from transformers import BertTokenizer
import torch
import torch.nn as nn


class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super(EmbeddingWithPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_len = max_len
        self.register_buffer('positional_encoding', self.create_positional_encoding(max_len, embedding_dim))
        # not a trainable parameter, but will be part of the model's state_dict.

    def create_positional_encoding(self, seq_len, d_model):
        pos_mat = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_mat[:, 0::2] = torch.sin(pos * div_term)
        pos_mat[:, 1::2] = torch.cos(pos * div_term)
        return pos_mat.unsqueeze(0) 

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        embeddings = self.token_embedding(input_ids) 
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(embeddings.device)
        return embeddings + positional_encoding


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "Hi da funda movane."

# tokens=tokenizer(text,return_tensors='pt',max_length=10,padding='max_length',truncation=True) #gives the output as pytoch tensor

# input_ids = tokens['input_ids']
# vocab_size = tokenizer.vocab_size #number of distinct tokens in the total tokens to create the map
# embedding_dim = 512
# #create embedding layer

# embedding_layer = EmbeddingWithPosEnc(vocab_size, embedding_dim, max_len=10)
# embedded_output = embedding_layer(input_ids)



# #adding positional encoding

# def positional_encoding(seq_len, d_model):
#     pos_mat=torch.zeros(seq_len, d_model)
#     pos=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
#     div=torch.exp(torch.arange(0,d_model,2).float() * ((torch.log(10000.0)))/d_model)
#     pos_mat[: ,0::2]=torch.sin(pos/div)
#     pos_mat[:, 1::2]=torch.cos(pos/div)
#     return pos_mat


# seq_len=input_ids.shape[1]
# pos_enc=positional_encoding(seq_len, embedding_dim)
# embedding_output=embedded_input+pos_enc.unsqueeze(0).to(embedded_input.device) 
#so basically we want the embedded input and the pos_enc to be in the same device like same cpu or sometimes gpu
#so it is safe to add so thats why bring the pos_enc to the same device as embedded_input
