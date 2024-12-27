import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from embedding import get_embedding  # 컬럼명을 하나씩 받아 임베딩을 생성하는 함수

# 입력 및 출력 경로 설정
input_path = './dataset/'      # CSV 파일들이 있는 디렉토리 (윈도우 경로는 슬래시 / 또는 역슬래시 \\ 사용 가능)
output_path = './columns_vector/'       # 임베딩을 저장할 디렉토리

# 출력 디렉토리가 존재하지 않으면 생성
os.makedirs(output_path, exist_ok=True)

# 입력 디렉토리 내 모든 파일 목록 가져오기
dataset = os.listdir(input_path)

# CSV 파일만 필터링
csv_files = [file for file in dataset if file.lower().endswith('.csv')]

# 각 CSV 파일을 처리
for csv_file in tqdm(csv_files, desc="CSV 파일 처리 중"):
    base_name = os.path.splitext(csv_file)[0]
    output_file = os.path.join(output_path, f"{base_name}.npy")
    
    # 이미 처리된 파일인지 확인
    if os.path.exists(output_file):
        print(f"{output_file} 파일이 이미 존재합니다. {csv_file}을 건너뜁니다.")
        continue  # 다음 파일로 넘어감
    
    csv_path = os.path.join(input_path, csv_file)
    
    try:
        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"{csv_file} 파일을 읽는 중 에러 발생: {e}")
        continue  # 에러 발생 시 다음 파일로 넘어감
    
    # 컬럼명 리스트 가져오기
    columns = list(df.columns)
    
    if not columns:
        print(f"{csv_file} 파일에 컬럼이 없습니다. 건너뜁니다.")
        continue  # 컬럼이 없으면 다음 파일로 넘어감
    
    embeddings = []
    
    # 각 컬럼명에 대해 임베딩 생성
    for column in tqdm(columns, desc=f"{csv_file}의 컬럼 임베딩 생성 중", leave=False):
        try:
            embedding = get_embedding(column)  # 컬럼명을 하나씩 전달
            embeddings.append(embedding)
        except Exception as e:
            print(f"{csv_file} 파일의 컬럼 '{column}' 임베딩 생성 중 에러 발생: {e}")
            embeddings.append([0] * 256)  # 에러 발생 시 임베딩을 0으로 채움 (예시)
    
    # 임베딩 리스트를 NumPy 배열로 변환
    embeddings_array = np.array(embeddings)
    
    # 임베딩 배열의 차원 확인
    if embeddings_array.ndim != 2:
        print(f"{csv_file} 파일의 임베딩 차원이 예상과 다릅니다: {embeddings_array.shape}. 건너뜁니다.")
        continue  # 2차원이 아니면 다음 파일로 넘어감
    
    try:
        # 임베딩 배열을 .npy 파일로 저장
        np.save(output_file, embeddings_array)
    except Exception as e:
        print(f"{csv_file} 파일의 임베딩을 저장하는 중 에러 발생: {e}")
        continue  # 저장 중 에러 발생 시 다음 파일로 넘어감
    
    print(f"{csv_file} 파일의 임베딩을 {output_file}에 저장했습니다.")
