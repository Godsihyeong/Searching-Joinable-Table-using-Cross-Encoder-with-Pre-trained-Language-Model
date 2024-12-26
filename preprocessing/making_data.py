import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast

# 경로 설정
columns_vector_path = './columns_vector/'  # 컬럼 임베딩이 저장된 폴더
embed_table_path = './embed_table/'        # 테이블 내용 임베딩이 저장된 폴더
input_table_path = './input_table/'        # 결합된 임베딩을 저장할 폴더

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(input_table_path, exist_ok=True)

# columns_vector와 embed_table 폴더 내 파일 목록 가져오기
columns_files = [f for f in os.listdir(columns_vector_path) if f.endswith('.npy')]
embed_table_files = [f for f in os.listdir(embed_table_path) if f.endswith('.csv')]

# 파일 이름(확장자 제외)으로 매칭하기 위해 기본 이름 추출
columns_base = {os.path.splitext(f)[0]: f for f in columns_files}
embed_table_base = {os.path.splitext(f)[0]: f for f in embed_table_files}

# 매칭되는 파일 이름 찾기 (두 폴더에 모두 존재하는 파일만)
common_tables = set(columns_base.keys()).intersection(set(embed_table_base.keys()))

if not common_tables:
    print("매칭되는 파일이 없습니다. 두 폴더에 동일한 이름의 파일이 있는지 확인해주세요.")
else:
    # 각 매칭된 테이블 처리
    for table_name in tqdm(common_tables, desc="테이블 결합 중"):
        # 각 파일의 전체 경로
        columns_file = os.path.join(columns_vector_path, columns_base[table_name])
        embed_table_file = os.path.join(embed_table_path, embed_table_base[table_name])

        try:
            # 컬럼 임베딩 로드 (.npy 파일)
            columns_embeddings = np.load(columns_file)  # Shape: (columns, embedding_dim)
        except Exception as e:
            print(f"{columns_base[table_name]} 파일을 로드하는 중 에러 발생: {e}")
            continue

        try:
            # 테이블 내용 임베딩 로드 (.csv 파일)
            embed_table_df = pd.read_csv(embed_table_file, header=None)
            rows, cols = embed_table_df.shape

            # 각 셀의 문자열을 리스트로 변환
            # 예를 들어, "[0.1, 0.2, ..., 0.256]" 형태의 문자열을 리스트로 변환
            embed_table_embeddings = embed_table_df.applymap(ast.literal_eval).values  # Shape: (rows, cols)

            # 각 셀을 numpy 배열로 변환하여 3D 배열 생성 (rows, cols, 256)
            embed_table_embeddings = np.array([
                [np.array(cell) for cell in row]
                for row in embed_table_embeddings
            ])  # Shape: (rows, columns, 256)

            # 임베딩 차원 확인 (256 차원)
            if embed_table_embeddings.shape[2] != 256:
                print(f"{embed_table_file} 파일의 임베딩 차원이 256이 아닙니다. 건너뜁니다.")
                continue

        except Exception as e:
            print(f"{embed_table_base[table_name]} 파일을 로드하는 중 에러 발생: {e}")
            continue

        # 컬럼 임베딩과 테이블 임베딩의 차원 확인
        columns, embedding_dim_columns = columns_embeddings.shape
        rows, cols, embedding_dim_table = embed_table_embeddings.shape

        if cols != columns:
            print(f"{table_name} 파일의 컬럼 수가 일치하지 않습니다. columns_vector: {columns}, embed_table: {cols}. 건너뜁니다.")
            continue

        if embedding_dim_columns != 256:
            print(f"{table_name} 파일의 컬럼 임베딩 차원이 256이 아닙니다: {embedding_dim_columns}. 건너뜁니다.")
            continue

        try:
            # 테이블 임베딩을 (columns, rows, 256)으로 전치
            embed_table_embeddings_transposed = np.transpose(embed_table_embeddings, (1, 0, 2))  # Shape: (columns, rows, 256)

            # 컬럼 임베딩을 (columns, 1, 256)으로 확장
            columns_embeddings_expanded = columns_embeddings[:, np.newaxis, :]  # Shape: (columns, 1, 256)

            # 컬럼 임베딩과 테이블 임베딩을 결합하여 (columns, rows + 1, 256)로 만듦
            combined_embeddings = np.concatenate((columns_embeddings_expanded, embed_table_embeddings_transposed), axis=1)  # Shape: (columns, rows + 1, 256)

        except Exception as e:
            print(f"{table_name} 파일의 임베딩을 결합하는 중 에러 발생: {e}")
            continue

        # 출력 파일 경로 설정 (.npy 파일로 저장)
        output_file = os.path.join(input_table_path, f"{table_name}.npy")

        try:
            # 결합된 임베딩을 .npy 파일로 저장
            np.save(output_file, combined_embeddings)
        except Exception as e:
            print(f"{table_name} 파일의 임베딩을 저장하는 중 에러 발생: {e}")
            continue

        print(f"{table_name} 파일의 임베딩을 {output_file}에 저장했습니다.")
