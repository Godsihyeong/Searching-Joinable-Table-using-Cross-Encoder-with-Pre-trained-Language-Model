import torch
from torch.utils.data import Dataset, DataLoader
import random

class TableDataset(Dataset):
    """
    각 샘플은 (table1, table2, label)이며,
    table1, table2는 각각 (cols, rows+1, embed_dim) 형태의 텐서.
    """
    def __init__(self, num_samples=1000, max_cols=8, min_cols=3, rows=10, embed_dim=256):
        super().__init__()
        self.samples = []
        for _ in range(num_samples):
            # 랜덤하게 cols1, cols2 지정
            cols1 = random.randint(min_cols, max_cols)
            cols2 = random.randint(min_cols, max_cols)

            table1 = torch.randn(cols1, rows+1, embed_dim)  # (cols1, rows+1, embed_dim)
            table2 = torch.randn(cols2, rows+1, embed_dim)  # (cols2, rows+1, embed_dim)

            label = random.randint(0, 1)
            self.samples.append((table1, table2, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    (table1, table2, label)을 묶어 batch를 생성.
    - 원본 cols 수를 별도로 real_cols1, real_cols2로 저장
    - 배치 내 최대 cols에 맞춰 패딩
    - table1s, table2s, labels, real_cols1, real_cols2 반환
    """
    table1s, table2s, labels = [], [], []
    real_cols1, real_cols2 = [], []

    max_cols_t1 = 0
    max_cols_t2 = 0

    for (t1, t2, lbl) in batch:
        max_cols_t1 = max(max_cols_t1, t1.size(0))  # t1.size(0) = cols1
        max_cols_t2 = max(max_cols_t2, t2.size(0))  # t2.size(0) = cols2
    
    for (t1, t2, lbl) in batch:
        cols1, rows1, emb1 = t1.shape  # (cols1, rows+1, embed_dim)
        cols2, rows2, emb2 = t2.shape  # (cols2, rows+1, embed_dim)

        # 저장: 실제 컬럼 수
        real_cols1.append(cols1)
        real_cols2.append(cols2)

        # 테이블1 패딩
        pad_cols1 = max_cols_t1 - cols1
        if pad_cols1 > 0:
            pad_tensor = torch.zeros(pad_cols1, rows1, emb1)
            t1 = torch.cat([t1, pad_tensor], dim=0)

        # 테이블2 패딩
        pad_cols2 = max_cols_t2 - cols2
        if pad_cols2 > 0:
            pad_tensor = torch.zeros(pad_cols2, rows2, emb2)
            t2 = torch.cat([t2, pad_tensor], dim=0)

        table1s.append(t1)  
        table2s.append(t2)  
        labels.append(lbl)
    
    table1s = torch.stack(table1s)  # (B, max_cols_t1, rows+1, emb)
    table2s = torch.stack(table2s)  # (B, max_cols_t2, rows+1, emb)
    labels = torch.tensor(labels)

    real_cols1 = torch.tensor(real_cols1)  # (B,)
    real_cols2 = torch.tensor(real_cols2)  # (B,)

    return table1s, table2s, labels, real_cols1, real_cols2

import torch
import torch.nn as nn

class VerticalSelfAttention(nn.Module):
    """
    - 입력: (B, max_cols, rows+1, E), 그리고 (B,)짜리 real_cols
    - 컬럼마다 row 시퀀스에 대해 MHA
    - self.rep_mode에 따라 CLS or MEAN
    - 출력: (B, max_cols, E)
    """
    def __init__(self, embed_dim=256, num_heads=4, rep_mode="cls"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rep_mode = rep_mode  # "cls" or "mean"

        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, real_cols):
        """
        x: (B, max_cols, R, E)
        real_cols: (B,) 실제 컬럼 개수
        """
        B, maxC, R, E = x.shape
        device = x.device

        rep_list = []

        for b_idx in range(B):
            # 실제 컬럼 수
            ncol = real_cols[b_idx].item()  # 파이썬 int

            table_b = x[b_idx]  # (maxC, R, E)

            reps_for_this_table = []
            for col_idx in range(ncol):
                col_tensor = table_b[col_idx]  # (R, E)

                # [CLS] + col_tensor
                cls_token_for_col = self.cls_token  # (1,1,E)
                col_tensor = col_tensor.unsqueeze(0)  # (1,R,E)
                col_with_cls = torch.cat([cls_token_for_col, col_tensor], dim=1)  # (1, R+1, E)

                # MHA
                attn_out, _ = self.mha(col_with_cls, col_with_cls, col_with_cls)
                out_ln = self.layernorm(attn_out)
                out_ffn = self.ffn(out_ln)
                out = out_ln + out_ffn

                if self.rep_mode == "cls":
                    rep_vec = out[:, 0, :]  # (1, E)
                else:  # "mean"
                    rep_vec = out.mean(dim=1)  # (1, E)

                reps_for_this_table.append(rep_vec)  # list of (1, E)

            if ncol > 0:
                reps_for_this_table = torch.cat(reps_for_this_table, dim=0)  # (ncol, E)
            else:
                # 만약 ncol=0이면(이론상), 1xE 0으로
                reps_for_this_table = torch.zeros(1, E).to(device)

            # 나머지 padding 컬럼에 해당하는 부분 (maxC - ncol) 만큼 0으로 채움
            if ncol < maxC:
                pad_cols = maxC - ncol
                pad_tensor = torch.zeros(pad_cols, E).to(device)
                reps_for_this_table = torch.cat([reps_for_this_table, pad_tensor], dim=0)

            # (maxC, E) -> (1, maxC, E)
            rep_list.append(reps_for_this_table.unsqueeze(0))

        reps = torch.cat(rep_list, dim=0)  # (B, maxC, E)
        return reps

from transformers import BertConfig, BertModel

class TableCrossEncoder(nn.Module):
    def __init__(self, 
                 pretrained_model_name="bert-base-uncased",
                 hidden_dim=256,
                 num_classes=2):
        super().__init__()
        
        config = BertConfig.from_pretrained(pretrained_model_name)
        config.position_embedding_type = 'none'  # 포지셔널 임베딩 제거
        self.bert = BertModel.from_pretrained(pretrained_model_name, config=config)

        self.projection = nn.Linear(256, 768)

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, table1_colreps, table2_colreps, real_cols1, real_cols2):
        """
        table1_colreps: (B, maxC1, E)
        table2_colreps: (B, maxC2, E)
        real_cols1, real_cols2: (B,) 실제 컬럼 수

        1) [CLS] + table1 + [SEP] + table2
        2) attention_mask: 실제 컬럼 위치는 1, 패딩은 0
        3) BERT forward -> cls_rep -> classifier
        """
        B, maxC1, E = table1_colreps.shape

        # ----- [1] 256 -> 768 변환 -----
        table1_colreps_768 = self.projection(table1_colreps)  # (B, maxC1, 768)
        table2_colreps_768 = self.projection(table2_colreps)  # (B, maxC2, 768)

        # ----- [2] [CLS], [SEP] 임베딩 (768차원) -----
        device = table1_colreps.device
        cls_token = torch.zeros(B, 1, 768).to(device)
        sep_token = torch.zeros(B, 1, 768).to(device)

        # sequence: (B, 1 + maxC1 + 1 + maxC2, E)
        sequence = torch.cat([cls_token, table1_colreps_768, sep_token, table2_colreps_768], dim=1)

        # attention_mask: same shape (B, seq_len)
        seq_len = sequence.size(1)
        attention_mask = torch.zeros(B, seq_len, dtype=torch.long).to(sequence.device)

        # 순서: [CLS] (1개) + table1 (maxC1개) + [SEP] (1개) + table2 (maxC2개)
        # 각 배치별로 실제 col만큼만 mask=1

        for b_idx in range(B):
            # CLS -> always 1
            attention_mask[b_idx, 0] = 1

            # table1 -> real_cols1[b_idx] 개
            r1 = real_cols1[b_idx].item()
            if r1 > 0:
                attention_mask[b_idx, 1 : 1 + r1] = 1

            # SEP -> 1
            sep_pos = 1 + maxC1
            attention_mask[b_idx, sep_pos] = 1

            # table2 -> real_cols2[b_idx] 개
            r2 = real_cols2[b_idx].item()
            if r2 > 0:
                attention_mask[b_idx, sep_pos+1 : sep_pos+1 + r2] = 1

        # ----- [3] BERT forward -----
        outputs = self.bert(
            inputs_embeds=sequence,
            attention_mask=attention_mask
        )
        cls_rep = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # ----- [4] Classifier -----
        logits = self.classifier(cls_rep)  # (B, num_classes)
        return logits
    
import torch
import torch.optim as optim
import torch.nn.functional as F

def train_model():
    # 하이퍼파라미터
    num_epochs = 2
    batch_size = 2
    lr = 1e-4
    
    # 1) 데이터셋 / 데이터로더
    train_dataset = TableDataset(num_samples=20, max_cols=8, min_cols=3, rows=10, embed_dim=256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    # 2) 모델 초기화
    vertical_attn = VerticalSelfAttention(embed_dim=256, num_heads=4, rep_mode="cls")
    cross_encoder = TableCrossEncoder(pretrained_model_name="bert-base-uncased", hidden_dim=256, num_classes=2)
    
    # 옵티마이저
    params = list(vertical_attn.parameters()) + list(cross_encoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    vertical_attn.train()
    cross_encoder.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for table1s, table2s, labels, real_cols1, real_cols2 in train_loader:
            # 2.1 VerticalSelfAttention -> (B, maxC1, E) / (B, maxC2, E)
            table1_reps = vertical_attn(table1s, real_cols1)  # (B, maxC1, E)
            table2_reps = vertical_attn(table2s, real_cols2)  # (B, maxC2, E)

            # 2.2 CrossEncoder(BERT) -> 로짓, mask까지 고려
            logits = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)

            # 2.3 Loss
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

train_model()