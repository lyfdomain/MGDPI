import random

import torch
import numpy as np
from torch import nn
import csv
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import *
import json
from torch.nn import functional
from collections import OrderedDict


class Autoencoder(nn.Module):
    def __init__(self, input_size, input_size2, hidden_size):
        super(Autoencoder, self).__init__()

    def forward(self, x, y):
        encoded = self.encoder(x)
        decoded = self.decoder(y)
        return encoded, decoded
    
    
def morgan_smiles(line, dim_num):
    mol = Chem.MolFromSmiles(line)
    feat = AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim_num)

    return feat

dat = "biosnap"
morgan_dim = 2048
if dat == "biosnap":
    model = torch.load("predti.pth")
    model.eval()
    
    jihe = "dataset/biosnap"
    drug_data = []

    dti = np.loadtxt('./'+ jihe +'/dti.txt').astype(dtype="int64")
    drug = np.loadtxt('./'+ jihe +'/drug.txt', dtype=str, comments=None)
    protein = np.loadtxt('./'+ jihe +'/protein.txt', dtype=str)
    ind = list(np.loadtxt("./"+ jihe +"/index_test.txt").astype(dtype="int64"))
    
    tt = []
    for i in range(len(drug)):
        xd = morgan_smiles(drug[i], morgan_dim)
        tt.append(xd)
    tt = np.array(tt)

    pp = np.loadtxt('./'+ jihe +'/bio_esm.txt', delimiter=",")

x = torch.from_numpy(tt).float()
y = torch.from_numpy(pp).float()

output1, output2 = model(x, y)
output1 = output1.detach().numpy()
output2 = output2.detach().numpy()

pre = cosine_similarity(output1, output2)

pr = []
tr = []
for i in range(len(ind)):
    pr.append(pre[ind[i][0], ind[i][1]])
    tr.append(dti[ind[i][0], ind[i][1]])

fpr, tpr, thresholds = sklearn.metrics.roc_curve(tr, pr)
area = sklearn.metrics.auc(fpr, tpr)
print("the Area Under the PRCurve is:", area)
# aucc.append(area)

aps = average_precision_score(tr, pr)
print("the AP score is:", aps)
# ap.append(aps)
