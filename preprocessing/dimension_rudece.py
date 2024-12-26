import os
import ast
import umap
import numpy as np
import pandas as pd
from tqdm import tqdm

# 입력 및 출력 경로 설정
path = './embed_additional_table/'          # CSV 파일들이 있는 디렉토리 (윈도우 경로는 슬래시를 / 또는 \\ 사용)
output_path = './reduced_table/'       # 차원 축소된 파일을 저장할 디렉토리

# 출력 디렉토리가 존재하지 않으면 생성
os.makedirs(output_path, exist_ok=True)

# 입력 디렉토리 내 모든 파일 목록 가져오기
dataset = os.listdir(path)

# UMAP 모델 초기화 (하이퍼파라미터는 필요에 따라 조정 가능)
umap_model = umap.UMAP(n_components=30, n_neighbors=10, random_state=35)

# 각 파일을 처리
for data in tqdm(dataset, desc='Processing'):
    # 파일의 기본 이름과 확장자 확인
    file_name, file_ext = os.path.splitext(data)
    
    # CSV 파일이 아닌 경우 건너뜀
    if file_ext.lower() != '.csv':
        print(f"Skipping {data} as it is not a CSV file.")
        continue
    
    # 출력 파일 이름 생성 ('reduced_' 접두사 추가)
    # 예: data = 'data1.csv' -> output_filename = 'reduced_a1.npy' (data[4:-4]가 'a1'일 경우)
    # 주의: data[4:-4]가 의도한 부분을 정확히 잘라내는지 확인 필요
    # 일반적으로 파일 이름 전체를 사용하고 싶다면, data[:-4]
    output_filename = f'reduced_{data[4:-4]}.npy'
    output_filepath = os.path.join(output_path, output_filename)
    
    # 이미 처리된 파일인지 확인
    if os.path.exists(output_filepath):
        print(f"{output_filename} already exists. Skipping {data}.")
        continue
    
    # CSV 파일의 전체 경로 생성
    csv_path = os.path.join(path, data)
    
    try:
        # CSV 파일을 DataFrame으로 읽기
        table = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {data}: {e}")
        continue
    
    # 행의 수가 30 미만인 경우 건너뜀
    if table.shape[0] < 30:
        print(f"Skipping {data} as it has less than 30 rows.")
        continue
    
    # 불필요한 컬럼 제거
    if 'Unnamed: 0' in table.columns:
        table = table.drop('Unnamed: 0', axis=1)
    
    if 'embedded_Unnamed: 0' in table.columns:
        table = table.drop('embedded_Unnamed: 0', axis=1)
    
    all_embeddings = []
    
    # 각 셀의 임베딩을 리스트로 변환
    for embedding in table.values.reshape(-1):
        if pd.notna(embedding):
            try:
                # 문자열을 리스트로 변환
                embedding_list = ast.literal_eval(embedding)
                all_embeddings.append(embedding_list)
            except Exception as e:
                print(f"Error parsing embedding in {data}: {e}")
                all_embeddings.append([0] * 256)  # 에러 발생 시 임베딩을 0으로 채움
        else:
            all_embeddings.append([0] * 256)      # NaN일 경우 임베딩을 0으로 채움
    
    # 임베딩을 NumPy 배열로 변환
    all_embeddings_np = np.array(all_embeddings)
    
    # 임베딩의 차원 확인
    if all_embeddings_np.shape[1] != 256:
        print(f"Embedding dimension mismatch in {data}: expected 256, got {all_embeddings_np.shape[1]}. Skipping.")
        continue
    
    try:
        # UMAP을 사용하여 차원 축소
        table_reduced = umap_model.fit_transform(all_embeddings_np)
    except Exception as e:
        print(f"Error during UMAP transformation for {data}: {e}")
        continue
    
    try:
        # 차원 축소된 데이터를 .npy 파일로 저장
        np.save(output_filepath, table_reduced)
    except Exception as e:
        print(f"Error saving reduced data for {data}: {e}")
        continue
    
    print(f"Saved reduced embeddings for {data} to {output_filepath}.")
