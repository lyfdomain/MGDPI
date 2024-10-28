import copy

import torch
import csv
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from model import *
import os

def morgan_smiles(line, dim_num):
    mol = Chem.MolFromSmiles(line)
    feat = AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim_num)

    return feat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(100)
jihe = "dataset/biosnap"
drug_data=[]

with open("./"+jihe+"/train.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        # 处理每一行数据
        drug_data.append(row)
drug_data= np.array(drug_data)

val_data = []
with open("./" + jihe + "/val.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        val_data.append(row)
val_data = np.array(val_data)

def nor(sim_mat):
    mm = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i,j]>0:
                mm[i, j] = (sim_mat[i, j]) / max(sim_mat[i, :])#(len(pp[j]))
            else:
                mm[i, j]=0
    return mm
dti = np.loadtxt('./'+jihe+'/dti.txt').astype(dtype="int64")
dti2 = np.loadtxt('./'+jihe+'/net_dti.txt').astype(dtype="int64")
SR = np.loadtxt('./'+jihe+'/DS.txt')
SP = np.loadtxt('./'+jihe+'/PS.txt')

def nor(sim_mat):
    mm = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i,j]>0:
                mm[i, j] = (sim_mat[i, j]) / max(sim_mat[i, :])#(len(pp[j]))
            else:
                mm[i, j]=0
    return mm

SP = nor(SP)
leraning_rate=0.0001
hidden_size = 512
num_epochs = 1000
morgan_dim = 1024
dru = np.loadtxt('./'+jihe+'/drug.txt', dtype=str, comments=None)
pro = np.loadtxt('./'+jihe+'/protein.txt', dtype=str)

drug=list(dru)
protein=list(pro)

pp=np.loadtxt('./'+jihe+'/bio_esm.txt', delimiter=",")

tt = []
for i in range(len(dru)):
    xd = morgan_smiles(dru[i], morgan_dim)
    tt.append(xd)
tt = np.array(tt)
ind_val = []
for i in range(len(val_data)):
    if val_data[i, 0] in drug and val_data[i, 1] in protein:
        ind_val.append([drug.index(val_data[i, 0]), protein.index(val_data[i, 1])])
tr = []
for i in range(len(ind_val)):
    tr.append(dti[ind_val[i][0], ind_val[i][1]])
auc_max=0
def test(e, f):
    output1 = e.detach().cpu().numpy()
    output2 = f.detach().cpu().numpy()
    pre = cosine_similarity(output1, output2)
    pr = []
    for i in range(len(ind_val)):
        pr.append(pre[ind_val[i][0], ind_val[i][1]])
    return pr
cof = np.zeros(dti.shape)
ind = []
for i in range(len(drug_data)):
    if drug_data[i,0] in drug and drug_data[i,1] in protein:
        ind.append([drug.index(drug_data[i,0]), protein.index(drug_data[i,1])])

for i in range(len(ind)):
    cof[ind[i][0], ind[i][1]] = 1

dti = torch.from_numpy(dti).float().cuda()
cof = torch.from_numpy(cof).float().cuda()
x = torch.from_numpy(tt).float().cuda()
y = torch.from_numpy(pp).float().cuda()
sr = torch.from_numpy(SR).float().cuda()
sp = torch.from_numpy(SP).float().cuda()
dti2 = torch.from_numpy(dti2).float().cuda()
input_size = len(tt[0])
input_size2 = len(pp[0])
autoencoder = Autoencoder(input_size, input_size2, hidden_size)
autoencoder = autoencoder.cuda()
model_max = copy.deepcopy(autoencoder)
criterion = nn.MSELoss()
criterion2 = AEFSLoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=leraning_rate)

for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()
    e, f = autoencoder(x, y)
    loss1 = criterion2(e, sr, f, sp, dti, cof, dti2)
    loss = 1 * loss1 
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.cpu().item()))

    autoencoder.eval()
    pr = test(e, f)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(tr, pr)
    area = sklearn.metrics.auc(fpr, tpr)
    aps = average_precision_score(tr, pr)
    if area > auc_max:
        auc_max = area
        model_max = copy.deepcopy(autoencoder)
    print(area, aps, "AUC_max", auc_max)

torch.save(model_max.cpu(), './predti.pth')





