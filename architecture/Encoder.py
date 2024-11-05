import torch
import torch.nn as nn
import numpy as np

from Attention import VerticalAttention, SelfAttention, MultiHeadAttention

class Encoder:
    def __init__(self, embed_size, heads, forward_expansion):
        self.multi_head_attention = MultiHeadAttention(embed_size, heads)
        
        self.norm1 = lambda x : (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
        self.norm2 = lambda x : (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
        
        self.W1 = np.random.randn(embed_size, embed_size * forward_expansion)
        self.W2 = np.random.randn(embed_size * forward_expansion, embed_size)
        self.b1 = np.zeros((1, embed_size * forward_expansion))
        self.b2 = np.zeros((1, embed_size))

    def point_wise_feed_forward_network(self, x):
        out = np.maximum(0, np.dot(x, self.W1) + self.b1)
        out = np.dot(out, self.W2) + self.b2
        
        return out

    def forward(self, x):
        attention = self.multi_head_attention.forward(x)
        x = self.norm1(x + attention)  # 잔차 연결과 정규화

        forward = self.point_wise_feed_forward_network(x)
        out = self.norm2(x + forward)  # 잔차 연결과 정규화
        
        return out
    

class CrossEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansioni, num_layers):
        self.encoders = [Encoder(embed_size, heads, forward_expansioni) for _ in range(num_layers)]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.sep_token = nn.Parameter(torch.randn(1, 1, embed_size))
        
    
    def forward(self, query_table, target_table):
        qry = [VerticalAttention(column).forward()[0] for column in query_table]
        trg = [VerticalAttention(column).forward()[0] for column in target_table]
        x = torch.cat
        
        
        
        for encoder in self.encoders:
            x = encoder.forward(x)
            
        cls_output = x[0]
        
        similarity = torch.sigmoid(self.linear(cls_output))
        
        return similarity