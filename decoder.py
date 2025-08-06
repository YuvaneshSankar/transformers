from input_embedding import EmbeddingWithPositionalEncoding
from maskedAttention import MaskedMultiHeadAttention
from addNnorm import AddAndNorm
from ffnn import FFNN
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, d_model, no_of_heads, dropout_rate=0.1, vocab_size=30522, max_len=512):
        super(Decoder, self).__init__()
        self.embedding_layer = EmbeddingWithPositionalEncoding(vocab_size, d_model, max_len)
        self.attention_layer = MaskedMultiHeadAttention(d_model, no_of_heads)
        self.add_norm1 = AddAndNorm(d_model)
        self.ffnn = FFNN(d_model, dropout_rate)
        self.add_norm2 = AddAndNorm(d_model)

    def forward(self, input_ids):
        x = self.embedding_layer(input_ids) 
        attn_out, attn_weights = self.attention_layer(x, input_ids)
        x = self.add_norm1(x, attn_out)
        #have to add cross attention here if needed
        #then a addnorm and ffnn then add norm then linear and softmax
        ffn_out = self.ffnn(x)
        x = self.add_norm2(x, ffn_out)
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

    decoder = Decoder(d_model, num_heads, dropout_rate=0.1, vocab_size=vocab_size, max_len=max_len)
    output, attn_weights = decoder(input_ids)

    print("Decoder output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
