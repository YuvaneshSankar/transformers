from addNnorm import AddAndNorm
from attention_layer import calculate_attention_scores
from input_embedding import EmbeddingWithPositionalEncoding
from ffnn import FFNN
import torch
import torch.nn as nn
from transformers import BertTokenizer


class encoder(nn.Module):
    def __init__(self,d_model, no_of_head, dropout_rate=0.1,vocab_size=30522, max_len=512):
        super(encoder,self).__init__()
        self.embedding_layer = EmbeddingWithPositionalEncoding(vocab_size, d_model, max_len)
        self.attenion_layer= calculate_attention_scores(d_model, no_of_head)
        self.add_norm1 = AddAndNorm(d_model)
        self.ffnn = FFNN(d_model, dropout_rate)
        self.add_norm2 = AddAndNorm(d_model)

    def forward(self, input_ids):
        x=self.embedding_layer(input_ids)
        attn_out,attn_weights=self.attenopn_layer(x)
        x=self.add_norm1(x, attn_out)
        ffnn_out=self.ffnn(x)
        x=self.add_norm2(x, ffnn_out)
        return x, attn_weights





if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "Hi da funda movane."
    tokens = tokenizer(text, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
    input_ids = tokens['input_ids']

    vocab_size = tokenizer.vocab_size
    d_model = 512
    num_heads = 8
    max_len = 10


    encoder = encoder(vocab_size, d_model, num_heads, max_len)
    output, attn_weights = encoder(input_ids)
    print("Encoder output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)