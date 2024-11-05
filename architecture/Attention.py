import numpy as np

class VerticalAttention:
    def __init__(self, x):
        self.x = x
        
        self.vert_W_q = np.random.randn(self.x.shape[1], self.x.shape[1])
        self.vert_W_k = np.random.randn(self.x.shape[1], self.x.shape[1])
        self.vert_W_v = np.random.randn(self.x.shape[1], self.x.shape[1])
        
    def forward(self):
        vert_Q = np.dot(self.x, self.vert_W_q)
        vert_K = np.dot(self.x, self.vert_W_k)
        vert_V = np.dot(self.x, self.vert_W_v)
        
        vertical_att_scores = np.dot(vert_Q, vert_K.T) / np.sqrt(vert_K.shape[1])
        vertical_att = self.softmax(vertical_att_scores)

        out = np.dot(vertical_att, vert_V)

        return out

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    
class SelfAttention:
    def __init__(self, embed_size, heads):
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size

        self.W_q = np.random.randn(self.embed_size, self.head_dim)
        self.W_k = np.random.randn(self.embed_size, self.head_dim)
        self.W_v = np.random.randn(self.embed_size, self.head_dim)

    def forward(self, x):
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        attention_scores = np.dot(Q, K.T) / np.sqrt(self.head_dim)
        attention = self.softmax(attention_scores)

        out = np.dot(attention, V)

        return out

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    

class MultiHeadAttention:
    def __init__(self, embed_size, heads):
        self.heads = heads
        self.head_dim = embed_size // heads
        self.self_attention = [SelfAttention(embed_size, heads) for _ in range(heads)]

    def forward(self, x):
        attention_outputs = [attention.forward(x) for attention in self.self_attention]
        out = np.concatenate(attention_outputs, axis=-1)
        return out