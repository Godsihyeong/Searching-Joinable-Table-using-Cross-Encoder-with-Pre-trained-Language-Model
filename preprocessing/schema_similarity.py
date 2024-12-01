import os
import numpy as np
import pandas as pd


# 1. column compatibility

compatibility_dict = {'object':{'object':1.0, 'float32':0.3, 'float64':0.3, 'int32':0.3, 'int64':0.3, 'bool':0.5},
                 'float32':{'object':0.3, 'float32':1.0, 'float64':1.0, 'int32':0.9, 'int64':0.9, 'bool':0.1},
                 'float64':{'object':0.3, 'float32':1.0, 'float64':1.0, 'int32':0.9, 'int64':0.9, 'bool':0.1},
                 'int32':{'object':0.3, 'float32':0.9, 'float64':0.9, 'int32':1.0, 'int64':1.0, 'bool':0.1},
                 'int64':{'object':0.3, 'float32':0.9, 'float64':0.9, 'int32':1.0, 'int64':1.0, 'bool':0.1},
                 'bool':{'object':0.5, 'float32':0.1, 'float64':0.1, 'int32':0.1, 'int64':0.1, 'bool':1.0}}

# input data dataframe.dtypes.tolist() or list(dataframe.dtypes)

def compatibility(df1_types, df2_types, compatibility_dict):
    
    compatibility_matrix = np.zeros((len(df1_types), len(df2_types)))
    
    for i, type1 in enumerate(df1_types):
        for j, type2 in enumerate(df2_types):
            compatibility_matrix[i, j] = compatibility_dict[str(type1)][str(type2)]
            
    return compatibility_matrix  # (query col length, target col length)

# 2. column name semantic similarity

# input data is embedding column (256-dim)

def cosine_similarity(embedding1, embedding2):
    norm1 = np.linalg.norm(embedding1, axis = 1, keepdims = True)
    norm2 = np.linalg.norm(embedding2, axis = 1, keepdims = True)
    
    normalized1 = embedding1 / norm1
    normalized2 = embedding2 / norm2
    
    col_similarity_matrix = np.dot(normalized1, normalized2.T)
    
    return col_similarity_matrix  # (query col length, target col length)

# 3. schema similarity

def schema_similarity(alpha, compatibility_matrix, col_similarity_matrix):
    return np.sum((alpha * compatibility_matrix) + ((1-alpha) * col_similarity_matrix))
    