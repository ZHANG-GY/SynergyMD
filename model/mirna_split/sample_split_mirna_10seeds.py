# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 16:24:08 2025

@author: 13700KF
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:47:48 2025

@author: 13700KF
"""

import numpy as np

# =================配置区域=================
# 输入文件路径 (确保是 [miRNA, Disease] 格式的二维数组)
INPUT_FILE = r'../../data/positive sample21213.npy'

OUTPUT_TRAIN_FILE = r'LOMOCV_mirna_train_10seeds.npy'
OUTPUT_TEST_FILE = r'LOMOCV_mirna_test_10seeds.npy'
# =========================================

# 1. 读取数据
pos_pairs = np.load(INPUT_FILE, allow_pickle=True)
all_mirnas = np.unique(pos_pairs[:, 0])
all_diseases = np.unique(pos_pairs[:, 1])
all_pos_set = set(map(tuple, pos_pairs))

print(f"Total Positive Pairs: {len(pos_pairs)}")
print(f"Unique miRNAs: {len(all_mirnas)} | Unique Diseases: {len(all_diseases)}")

# --- 通用负样本生成函数 (显式指定候选列表) ---
def generate_negs(candidate_mirnas, candidate_diseases, pos_set, count):
    """
    candidate_mirnas: 允许抽样的miRNA列表
    candidate_diseases: 允许抽样的疾病列表
    """
    neg_set = set()
    while len(neg_set) < count:
        # 每次多生成一些以抵消冲突
        batch_size = int((count - len(neg_set)) * 1.5) + 10  
        
        batch_m = np.random.choice(candidate_mirnas, batch_size)
        batch_d = np.random.choice(candidate_diseases, batch_size)
        
        # 组合成对 (miRNA, Disease)
        candidates = set(zip(batch_m, batch_d))
        
        # 过滤掉正样本和已存在的负样本
        valid_negs = candidates - pos_set - neg_set
        neg_set.update(valid_negs)
        
    return list(neg_set)[:count]

# --- 初始化容器 ---
dict_train_all = {}
dict_test_all = {}

# --- 2. 循环 10 次划分 ---
for seed in range(10):
    np.random.seed(seed)
    
    # A. 划分实体 (10% miRNA 作为测试集)
    test_mirnas = np.random.choice(all_mirnas, int(len(all_mirnas) * 0.1), replace=False)
    # 剩下的为训练 miRNA
    train_mirnas = np.setdiff1d(all_mirnas, test_mirnas)
    
    # B. 划分正样本 (注意：这里检查的是第0列 miRNA)
    mask_test = np.isin(pos_pairs[:, 0], test_mirnas)
    
    test_pos = pos_pairs[mask_test]
    train_pos = pos_pairs[~mask_test]
    
    # C. 生成负样本
    # 训练集负样本：miRNA只能来自 [训练集miRNA]，疾病可以是任意疾病
    train_neg = generate_negs(train_mirnas, all_diseases, all_pos_set, len(train_pos))
    
    # 测试集负样本：miRNA只能来自 [测试集miRNA]，疾病可以是任意疾病
    # (这样保证了测试集里全是“新miRNA”的样本)
    test_neg = generate_negs(test_mirnas, all_diseases, all_pos_set, len(test_pos))
    
    # D. 组合并打乱 [miRNA, Disease, Label]
    train_data = np.vstack([
        np.column_stack([train_pos, np.ones(len(train_pos))]),
        np.column_stack([train_neg, np.zeros(len(train_neg))])
    ])
    test_data = np.vstack([
        np.column_stack([test_pos, np.ones(len(test_pos))]),
        np.column_stack([test_neg, np.zeros(len(test_neg))])
    ])
    
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    # E. 存入字典
    dict_train_all[seed] = train_data
    dict_test_all[seed] = test_data
    
    print(f"Seed {seed}: Train {len(train_data)} | Test {len(test_data)} (Test miRNAs: {len(test_mirnas)})")

# --- 3. 保存文件 ---
np.save(OUTPUT_TRAIN_FILE, dict_train_all)
np.save(OUTPUT_TEST_FILE, dict_test_all)

print("\nSaved successfully!")
print(f"Train file: {OUTPUT_TRAIN_FILE}")
print(f"Test file : {OUTPUT_TEST_FILE}")
