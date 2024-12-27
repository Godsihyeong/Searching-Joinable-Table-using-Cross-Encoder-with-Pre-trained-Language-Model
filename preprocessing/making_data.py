import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

# 경로 설정
columns_vector_path = './columns_vector/'  # 컬럼 임베딩이 저장된 폴더
embed_table_path = './embed_table/'        # 테이블 내용 임베딩이 저장된 폴더
output_path = './output_npy/'              # 결합된 결과를 저장할 폴더

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(output_path, exist_ok=True)

table_embeddings = os.listdir(embed_table_path)     # csv
columns_vectors = os.listdir(columns_vector_path)   # npy

# 안전한 파싱 함수
def safe_literal_eval(cell):
    try:
        if pd.isna(cell):  # NaN 값 처리
            return [0] * 256  # 기본값 반환
        return ast.literal_eval(cell)  # 파싱 시도
    except (ValueError, SyntaxError):
        return [0] * 256  # 잘못된 값은 기본값 반환

for table in tqdm(table_embeddings, desc='Processing...'):

    output_file = os.path.join(output_path, f"{table[4:-4]}.npy")
    
    # CSV 파일 로드
    embed_table_df = pd.read_csv(os.path.join(embed_table_path, table))

    # 각 셀을 안전하게 파싱하고 NumPy 배열로 변환
    embed_table_embeddings = embed_table_df.applymap(safe_literal_eval).values  # 각 셀을 리스트로 변환
    embed_table_embeddings = np.array([
        [np.array(cell) for cell in row]  # 각 셀을 NumPy 배열로 변환
        for row in embed_table_embeddings
    ])  # 결과: (rows, cols, list_dim)

    embed_table_embeddings_transposed = np.transpose(embed_table_embeddings, (1, 0, 2))

    columns_file = table[4:-4] + '.npy'
    columns_embeddings = np.load(os.path.join(columns_vector_path, columns_file))
    columns_embeddings_expanded = columns_embeddings[:, np.newaxis, :]
    
    combined_embeddings = np.concatenate((embed_table_embeddings_transposed, columns_embeddings_expanded), axis=1)

    # 결합된 결과 저장
    np.save(output_file, combined_embeddings)
    print(f"{table}.npy 파일이 {output_path}에 저장되었습니다.")
