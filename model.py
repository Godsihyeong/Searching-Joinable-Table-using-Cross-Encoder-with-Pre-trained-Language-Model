import torch.nn as nn
import torch

from Attention import VerticalAttention
from Encoder import Encoder

class CrossEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, num_layers):
        super(CrossEncoder, self).__init__()
        self.encoders = nn.ModuleList([Encoder(embed_size, heads, forward_expansion) for _ in range(num_layers)])
        self.segment_embeddings = nn.Embedding(2, embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.sep_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.linear = nn.Linear(embed_size, 1)

    def forward(self, query_table, target_table):
        qry = torch.stack([VerticalAttention(column).forward()[0].unsqueeze(0) for column in query_table], dim=0)
        trg = torch.stack([VerticalAttention(column).forward()[0].unsqueeze(0) for column in target_table], dim=0)

        cls = self.cls_token
        sep = self.sep_token

        input_seq = torch.cat([cls, qry, sep, trg, sep], dim=0)

        segment_ids = torch.tensor(
            [0] * (1 + qry.size(0) + 1) + [1] * (trg.size(0) + 1), dtype=torch.long
        ).unsqueeze(0)
        segment_embedding = self.segment_embeddings(segment_ids)  # (1, seq_len, embed_size)

        # input_seq와 차원 맞추기 (broadcasting 사용)
        segment_embedding = segment_embedding.reshape(segment_embedding.size(1), segment_embedding.size(0), segment_embedding.size(2))  # (10, seq_len, embed_size)
        
        x = input_seq + segment_embedding.permute(1, 0, 2) # (batch, seq_len, embed_size)

        for encoder in self.encoders:
            x = encoder.forward(x)

        cls_output = x[:, 0, :]
        similarity = torch.sigmoid(self.linear(cls_output))

        return similarity