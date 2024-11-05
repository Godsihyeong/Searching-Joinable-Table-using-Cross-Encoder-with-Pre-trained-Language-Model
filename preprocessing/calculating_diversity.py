import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from diversity_score import clustering, diversity_score

path = './reduced_table/'
npy_datasets = os.listdir(path)
print(len(npy_datasets))

results = {'Query': [], 'Target': [], 'Diversity': []}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in tqdm(range(len(npy_datasets)), desc="Outer Loop Progress"):
    for j in tqdm(range(i + 1, len(npy_datasets)), desc="Inner Loop Progress"):
        
        # npy 파일을 불러와 GPU에 올림
        qry = torch.tensor(np.load(path + npy_datasets[i]), device=device)
        trg = torch.tensor(np.load(path + npy_datasets[j]), device=device)

        # clustering과 diversity_score 함수가 GPU에서 작동하도록 수정 필요
        cluster_labels = clustering(qry, trg)  # clustering 함수에 GPU 코드 필요
        div_score = diversity_score(cluster_labels)  # diversity_score 함수에 GPU 코드 필요

        results['Query'].append(npy_datasets[i])
        results['Target'].append(npy_datasets[j])
        results['Diversity'].append(div_score)

data = pd.DataFrame(results)

data.to_csv('./diversity_dataframe.csv')