# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:47:48 2025

@author: 13700KF
"""

import numpy as np

# =================配置区域=================
# 输入文件路径
INPUT_FILE = r'../../data/sample/positive sample21213.npy'

# 输出文件路径
OUTPUT_TRAIN_FILE = r'LODOCV_train_disease_10seeds.npy'
OUTPUT_TEST_FILE = r'LODOCV_test_disease_10seeds.npy'
# =========================================

# 1. 读取数据
pos_pairs = np.load(INPUT_FILE, allow_pickle=True)
all_mirnas = np.unique(pos_pairs[:, 0])
all_diseases = np.unique(pos_pairs[:, 1])
all_pos_set = set(map(tuple, pos_pairs))

print(f"Total Positive Pairs: {len(pos_pairs)}")

# --- 负样本生成函数 (严格去重) ---
def generate_strict_negatives(target_diseases, all_mirnas, pos_set, count):
    neg_set = set()
    while len(neg_set) < count:
        batch_size = int((count - len(neg_set)) * 1.5) + 10  
        batch_m = np.random.choice(all_mirnas, batch_size)
        batch_d = np.random.choice(target_diseases, batch_size)
        candidates = set(zip(batch_m, batch_d))
        valid_negs = candidates - pos_set - neg_set
        neg_set.update(valid_negs)
    return list(neg_set)[:count]

# --- 初始化容器 ---
# 我们使用字典来存储，Key是随机种子(0-9)，Value是对应的数据集数组
dict_train_all = {}
dict_test_all = {}

# --- 2. 循环 10 次划分 ---
for seed in range(10):
    np.random.seed(seed)
    
    # A. 划分实体 (10% 疾病作为测试集)
    test_diseases = np.random.choice(all_diseases, int(len(all_diseases) * 0.1), replace=False)
    # 剩下的为训练疾病 (setdiff1d 自动去重并排序，安全)
    train_diseases = np.setdiff1d(all_diseases, test_diseases)
    
    # B. 划分正样本
    mask_test = np.isin(pos_pairs[:, 1], test_diseases)
    test_pos = pos_pairs[mask_test]
    train_pos = pos_pairs[~mask_test]
    
    # C. 生成负样本
    # 训练集负样本：只能来自 [训练疾病]
    train_neg = generate_strict_negatives(train_diseases, all_mirnas, all_pos_set, len(train_pos))
    # 测试集负样本：只能来自 [测试疾病]
    test_neg = generate_strict_negatives(test_diseases, all_mirnas, all_pos_set, len(test_pos))
    
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
    
    print(f"Seed {seed}: Train {len(train_data)} | Test {len(test_data)} (Test Diseases: {len(test_diseases)})")

# --- 3. 保存文件 ---
# 注意：保存字典时必须使用 allow_pickle=True
np.save(OUTPUT_TRAIN_FILE, dict_train_all)
np.save(OUTPUT_TEST_FILE, dict_test_all)

print("\nSaved successfully!")
print(f"Train file: {OUTPUT_TRAIN_FILE}")
print(f"Test file : {OUTPUT_TEST_FILE}")
