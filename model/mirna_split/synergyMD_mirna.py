# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:29:30 2024

@author: 13700KF
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import optuna
from tqdm import tqdm

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)

torch.cuda.manual_seed(4)
torch.cuda.manual_seed_all(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transfer = StandardScaler()

train_data_dict  = np.load(r'LOMOCV_mirna_train_10seeds.npy',allow_pickle=True).item()
test_data_dict  = np.load(r'LOMOCV_mirna_test_10seeds.npy',allow_pickle=True).item()

skf1 = KFold(n_splits=5,shuffle=True)
mir_gcn_fea = np.load("../../data/miRNA graph features.npy")
mir_seq_fea = np.load("../../data/miRNA sequence features.npy")
mir_seq_fea = transfer.fit_transform(mir_seq_fea)
with open("../../data/miRNA sequence names.pkl",'rb') as f:
    mir_name = pickle.load(f)
    
with open("../../data/miRNA graph names.pkl",'rb') as f:
    miRNA_name_1602 = pickle.load(f)
mir_gcn_fea = mir_gcn_fea[:1602]

dis_gcn_fea = np.load("../../data/disease graph features.npy")
dis_gcn_fea = dis_gcn_fea[:7732]
dis_gcn_fea = transfer.fit_transform(dis_gcn_fea)

with open("../../data/disease graph names.pkl",'rb') as f:
    mesh_name_7732 = pickle.load(f)
with open("../../data/disease text names.pkl",'rb') as f:
    mesh_name_7053 = pickle.load(f)
dis_txt_fea = np.load("../../data/disease text features.npy")

dis_gcn_idx = [mesh_name_7732.index(i) for i in mesh_name_7053]
dis_gcn_fea = dis_gcn_fea[dis_gcn_idx]

mir_seq_idx = [mir_name.index(i) for i in miRNA_name_1602]
mir_seq_fea = mir_seq_fea[mir_seq_idx]



transfer = StandardScaler()
dis_txt_fea = transfer.fit_transform(dis_txt_fea)


mir_gcn_fea = transfer.fit_transform(mir_gcn_fea)






def data_iter(batch_size, fea1, fea2):
    num_examples = len(fea1)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield fea1[batch_indices], fea2[batch_indices]

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def get_logits( image_features, text_features, logit_scale):
    # 计算image_features @ text_features.T相似度矩阵
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T
    return logits_per_image, logits_per_text
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()      

class MLP1(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2,hidden_channels1, hidden_channels2, hidden_channels3, hidden_channels4,out_channels, dropout):
        super(MLP1, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels1, hidden_channels1)
        self.linear2 = torch.nn.Linear(in_channels2, hidden_channels2)
        self.relu = torch.nn.ReLU()
        self.dropout = dropout
        self.linear3 = torch.nn.Linear(hidden_channels1, hidden_channels3)
        self.linear4 = torch.nn.Linear(hidden_channels2, hidden_channels4)
        self.linear5 = torch.nn.Linear(hidden_channels3, out_channels)
        self.linear6 = torch.nn.Linear(hidden_channels4, out_channels)
    def forward(self, x1, x2, normalize: bool = False):
        feature1 = self.linear1(x1)
        feature2 = self.linear2(x2)
        feature1 = self.relu(feature1)
        feature2 = self.relu(feature2)
        feature1 = F.dropout(feature1, self.dropout, training=self.training)
        feature2 = F.dropout(feature2, self.dropout, training=self.training)
        feature1 = self.linear3(feature1)
        feature2 = self.linear4(feature2)
        feature1 = self.relu(feature1)
        feature2 = self.relu(feature2)
        feature1 = F.dropout(feature1, self.dropout, training=self.training)
        feature2 = F.dropout(feature2, self.dropout, training=self.training)
        feature1 = self.linear5(feature1)
        feature2 = self.linear6(feature2)
        # s1 = F.normalize(feature1, dim=-1) if normalize else feature1
        # s2 = F.normalize(feature2, dim=-1) if normalize else feature2
        s1 = feature1 / feature1.norm(dim=1, keepdim=True)
        s2 = feature2 / feature2.norm(dim=1, keepdim=True)
        return s1,s2
    
    
class MLP2(torch.nn.Module):
    def __init__(self, in_channels1, hidden_channels1,  hidden_channels2, out_channels, dropout):
        super(MLP2, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels1, hidden_channels1)
        self.linear2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.relu = torch.nn.ReLU()
        self.dropout = dropout
        self.linear3 = torch.nn.Linear(hidden_channels2, out_channels)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        feature1 = self.linear1(x1)
        feature1 = self.relu(feature1)
        feature1 = F.dropout(feature1, self.dropout, training=self.training)
        feature1 = self.linear2(feature1)
        feature1 = self.relu(feature1)
        feature1 = F.dropout(feature1, self.dropout, training=self.training)
        feature1 = self.linear3(feature1)
        feature1 = self.sigmoid(feature1)
        return feature1

    
    
fenlei_loss = nn.BCELoss()
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
sigmoid = torch.nn.Sigmoid()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

dis_txt_fea = torch.tensor(dis_txt_fea).to(torch.float32).to(device)
dis_gcn_fea = torch.tensor(dis_gcn_fea).to(torch.float32).to(device)
mir_seq_fea = torch.tensor(mir_seq_fea).to(torch.float32).to(device)
mir_gcn_fea = torch.tensor(mir_gcn_fea).to(torch.float32).to(device)


cl1 = MLP1(128,768,512,512,256,256,256,0).to(device)
cl2 = MLP1(128,120,512,512,256,256,256,0).to(device)
optimizer1 = torch.optim.Adam(cl1.parameters(), lr=0.0001, weight_decay=0.001)
optimizer2 = torch.optim.Adam(cl2.parameters(), lr=0.0001, weight_decay=0.00001)
cl_loss1 = []
cl_loss2 = []
MLP_train_loss = []
MLP_val_loss = []
train_auc_list = []
train_acc_list = []
val_auc_list = []
val_acc_list = []

y_true_list = []
y_score_list = []

for iteration in tqdm(range(80)):
    loss1_list = []
    
    cl1.train()
    
    
    for fea1,fea2 in data_iter(600, dis_gcn_fea, dis_txt_fea):
    
        
        z1, z2 = cl1(fea1, fea2)
        logits_per_txt, logits_per_gcn = get_logits(z1, z2, logit_scale)
        ground_truth = torch.arange(logits_per_txt.shape[0],dtype=torch.long).to(device)
        loss1 = (loss_img(logits_per_gcn,ground_truth) + loss_txt(logits_per_txt,ground_truth))/2
        loss1_list.append(loss1.item())
        optimizer1.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer1.step()
    final_loss1 = sum(loss1_list)/len(loss1_list)
    cl_loss1.append(final_loss1)
for iteration in tqdm(range(100)):
    loss2_list = []
    cl2.train()
    for fea1,fea2 in data_iter(300, mir_gcn_fea, mir_seq_fea):    
        
        z1,z2 = cl2(fea1,fea2)
        logits_per_txt, logits_per_gcn = get_logits(z1, z2, logit_scale)
        ground_truth = torch.arange(logits_per_txt.shape[0],dtype=torch.long).to(device)
        loss2 = (loss_img(logits_per_gcn,ground_truth) + loss_txt(logits_per_txt,ground_truth))/2
        loss2_list.append(loss2.item())
        optimizer2.zero_grad()
        loss2.backward(retain_graph=True)
        optimizer2.step()
    final_loss2 = sum(loss2_list)/len(loss2_list)
    cl_loss2.append(final_loss2)
    
cl1.eval()
dis_gcn_fea1,dis_txt_fea1 = cl1(dis_gcn_fea,dis_txt_fea)

cl2.eval()
mir_gcn_fea1,mir_seq_fea1 = cl2(mir_gcn_fea,mir_seq_fea)

dis_gcn_fea1 = dis_gcn_fea1.detach()
dis_txt_fea1 = dis_txt_fea1.detach()
mir_gcn_fea1 = mir_gcn_fea1.detach()
mir_seq_fea1 = mir_seq_fea1.detach()

dis_fea = torch.concatenate((dis_gcn_fea1,dis_txt_fea1),axis=1)
# dis_gcn_fea2 = transfer.fit_transform(dis_gcn_fea1.detach().cpu().numpy())
# dis_txt_fea2 = transfer.fit_transform(dis_txt_fea1.detach().cpu().numpy())
mir_fea = torch.concatenate((mir_gcn_fea1,mir_seq_fea1),axis=1)
# mir_gcn_fea2  = transfer.fit_transform(mir_gcn_fea1.detach().cpu().numpy())
# mir_seq_fea2  = transfer.fit_transform(mir_seq_fea1.detach().cpu().numpy())
# dis_fea = 0.5*dis_gcn_fea2 + 0.5*dis_txt_fea2
# mir_fea = 0.5*mir_gcn_fea2 + 0.5*mir_seq_fea2
# dis_fea = transfer.fit_transform(dis_fea.detach().cpu().numpy())
# mir_fea = transfer.fit_transform(mir_fea.detach().cpu().numpy())
# dis_fea = torch.tensor(dis_fea).to(device)
# mir_fea = torch.tensor(mir_fea).to(device)


final_train_loss_list = []
final_train_acc_list = []
final_train_auc_list = []
final_train_pre_list = []
final_train_f1_list = []
final_train_rec_list = []

final_test_acc_list = []
final_test_auc_list = []
final_test_pre_list = []
final_test_f1_list = []
final_test_rec_list = []




train_acc_all = []
test_acc_all = []
train_auc_all = []
test_auc_all = []

fpr_list = []
tpr_list = []
precision_list = []
recall_list = []
ap_lsit = []
fold=1


for seed in range(10):
    train_samples = train_data_dict[seed]
    test_samples  = test_data_dict[seed]
    train_label = train_samples[:,2].astype(float)
    test_label = test_samples[:,2].astype(float)
    
    
    
    test_acc_list = []
    test_auc_list = []
    test_f1_list = []
    test_rec_list = []
    test_pre_list = []
    
    # for train_index, val_index in skf1.split(train_sample, train_label):
    MLP_fenlei = MLP2(1024,512,64,1,0.8).to(device)
    optimizer_MLP_fenlei = torch.optim.Adam(MLP_fenlei.parameters(),lr=0.0001,weight_decay=0.002)
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_MLP_fenlei, step_size=20, gamma=1)
    train_loss_list = []
    train_acc_list = []
    train_auc_list = []
    train_pre_list = []
    train_f1_list = []
    train_rec_list = []
    train_label = torch.tensor(train_label).to(device)
    t_mir_idx = [miRNA_name_1602.index(i[0]) for i in train_samples] 
    t_dis_idx = [mesh_name_7053.index(i[1]) for i in train_samples]
    t_sample_fea = torch.concatenate((mir_fea[t_mir_idx],dis_fea[t_dis_idx]),axis=1).to(device)
    train_data = TensorDataset(t_sample_fea,train_label)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=False)
    test_mir_idx = [miRNA_name_1602.index(i[0]) for i in test_samples] 
    test_dis_idx = [mesh_name_7053.index(i[1]) for i in test_samples]
    
    test_sample_fea = torch.concatenate((mir_fea[test_mir_idx],dis_fea[test_dis_idx]),axis=1).to(device)
    
    for eopch in tqdm(range(400)):
        MLP_fenlei.train()
        loss3_list = []
        for x_batch, y_batch in train_loader:
            pred_y = MLP_fenlei(x_batch)
           
            loss3 = fenlei_loss(pred_y.reshape(-1), y_batch.to(torch.float32))
            loss3_list.append(loss3.item())
            optimizer_MLP_fenlei.zero_grad()
            loss3.backward(retain_graph=True)
            optimizer_MLP_fenlei.step()
            scheduler_encoder.step()
            
        final_loss3 = sum(loss3_list)/len(loss3_list)
            # acc1 += torch.eq(y_pred, y_batch).int().sum()
            # all_y_pred = all_y_pred + y_pred.cpu().numpy().tolist()
        train_loss_list.append(final_loss3)
        MLP_fenlei.eval()    
        pred_y2 = MLP_fenlei(t_sample_fea)
        y_pred = pred_y2.detach().cpu().numpy().tolist()
        y_pred2 = []
        for i in y_pred:
            if i[0]>=0.5:
                y_pred2.append(1)
            else:
                y_pred2.append(0)
                
        train_acc = accuracy_score(train_label.cpu().numpy().tolist(),y_pred2)
        auc1 = roc_auc_score(train_label.cpu().numpy().tolist(), pred_y2.detach().cpu().numpy())
        pre1 = precision_score(train_label.cpu().numpy().tolist(),y_pred2)
        f11 = f1_score(train_label.cpu().numpy().tolist(),y_pred2)
        rec1 = recall_score(train_label.cpu().numpy().tolist(),y_pred2)
        train_acc_list.append(train_acc)
        train_auc_list.append(auc1)
        train_pre_list.append(pre1)
        train_f1_list.append(f11)
        train_rec_list.append(rec1)
    
    
        
        
      
    final_train_loss_list.append(train_loss_list)
    final_train_acc_list.append(train_acc_list)
    final_train_auc_list.append(train_auc_list)
    final_train_pre_list.append(train_pre_list)
    final_train_f1_list.append(train_f1_list)
    final_train_rec_list.append(train_rec_list)
    
    
    MLP_fenlei.eval()
    
    with torch.no_grad():
        pred_y = MLP_fenlei(test_sample_fea)
        y_pred = pred_y.detach().detach().cpu().numpy().tolist()
        
        y_pred2 = []
        for i in y_pred:
            if i[0]>=0.5:
                y_pred2.append(1)
            else:
                y_pred2.append(0)
    y_true_list.append(test_label.tolist())
    y_score_list.append(pred_y.detach().cpu().numpy())
    test_acc = accuracy_score(test_label.tolist(),y_pred2)
    auc3 = roc_auc_score(test_label.tolist(), pred_y.detach().cpu().numpy())
    pre3 = precision_score(test_label.tolist(),y_pred2)
    f13 = f1_score(test_label.tolist(),y_pred2)
    rec3 = recall_score(test_label.tolist(),y_pred2)
    test_acc_list.append(test_acc)
    test_auc_list.append(auc3)
    test_pre_list.append(pre3)
    test_f1_list.append(f13)
    test_rec_list.append(rec3)
    
    train_acc_all.append(train_acc_list[-1])
    train_auc_all.append(train_auc_list[-1])
    test_acc_all.append(test_acc)
    test_auc_all.append(auc3)
    final_test_acc_list.append(test_acc_list)
    final_test_auc_list.append(test_auc_list)
    final_test_pre_list.append(test_pre_list)
    final_test_f1_list.append(test_f1_list)
    final_test_rec_list.append(test_rec_list)

