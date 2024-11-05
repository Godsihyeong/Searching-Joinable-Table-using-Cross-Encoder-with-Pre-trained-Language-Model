import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import os

client = OpenAI(api_key='your_api_key')

# get embedding vector (256-dim) for text data in tabular data
def get_embedding(text, model="text-embedding-3-small"):
    if type(text) != str:
        text = str(text)
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding[:256]

def table_embedding(table_path):
    try:
        table = pd.read_csv(table_path, encoding='cp949')
    except:
        table = pd.read_csv(table_path)

    column_length = len(table.columns.tolist())

    # 테이블 샘플링 (300개 이상의 데이터인 경우)
    if len(table) > 300:
        table = table.sample(n=300, random_state=42)

    # 열 임베딩 진행
    for i, column in enumerate(table.columns):
        tqdm.pandas(desc=f'Processing embedding for col_{i}')

        try:
            # 각 열에 대해 임베딩 수행
            table[f'embedded_{column}'] = table[column].progress_apply(
                lambda x: get_embedding(x, model='text-embedding-3-small')
            )
        except Exception as e:
            print(f"Error occurred in table '{table_path}' for column '{column}': {e}")
            print(f"Skipping table '{table_path}'...")
            return None  # 테이블 중단 후 None 반환

    print(f'Completed Table Embedding for {table_path}')
    return table.iloc[:, column_length:]


