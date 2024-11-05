import os
import pandas as pd
import numpy as np

import umap
import ast
import hdbscan

def clustering(qry, trg):
    
    print()
    print('='*10 + 'Clustering' + '='*10)
    print()
    
    combined = np.vstack((qry, trg))
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size = 10)
    cluster_labels = clusterer.fit_predict(combined)
    
    return cluster_labels

def diversity_score(cluster_labels):
    total_samples = len(cluster_labels)
    noise_samples = np.sum(cluster_labels == -1)
    valid_labels = cluster_labels[cluster_labels != -1]
            
    cluster_counts = np.bincount(valid_labels)
    print(f'Cluster Counts : {cluster_counts}')

    noise_ratio = noise_samples / total_samples
    print(f'Noise Ratio : {noise_ratio:.4f}')
    print()
    
    return (len(cluster_counts) / total_samples) * (1 - noise_ratio)