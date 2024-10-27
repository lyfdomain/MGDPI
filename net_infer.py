import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics
from sklearn.metrics import *
import copy

prenum=2181
jihe="dataset/biosnap"

def reconstruct(S):
    xx=len(S)
    SS=copy.deepcopy(S)
    sorted_mol = (SS).argsort(axis=1).argsort(axis=1)
    np.fill_diagonal(sorted_mol, 0)
    sorted_mol=(xx-1)*np.ones((xx,xx))-sorted_mol
    sorted_mol[sorted_mol == 0] = 1
    sorted_mol = 1/((sorted_mol))
    sorted_mol = (sorted_mol+sorted_mol.T)/2
    return sorted_mol

def drug_rowsimm(dd):
    mm = np.zeros(dd.shape)
    for i in range(dd.shape[0]):
        # for j in range(dd.shape[1]):
            if np.sum(dd[i, :])!=0:
               mm[i, :] = dd[i, :] / np.sum(dd[i, :])
    mm[mm < 0] = 0
    return mm


def dis_colsimm(dd):
    mm = np.zeros(dd.shape)
    for j in range(dd.shape[1]):
            if np.sum(dd[:, j]) != 0:
                mm[:, j] = dd[:, j] / np.sum(dd[:, j])
    mm[mm < 0] = 0
    return mm

A_real = np.loadtxt("./"+jihe+"/dti.txt")  
A = np.loadtxt("./"+jihe+"/dti_train.txt")

SR = np.loadtxt("./"+jihe+"/DS.txt")
SP = np.loadtxt("./"+jihe+"/PS.txt")
SR_1 = reconstruct(SR)
SP_1 = reconstruct(SP)
SR_1=drug_rowsimm(SR_1*SR)
SP_1=dis_colsimm(SP_1*SP)
A_pred = 0.5*np.matmul(A,SP_1) + 0.5*np.matmul(SR_1,A)

for i in range(len(A)):
    for j in range(len(A[0])):
        A_pred[i,j] = max(A[i,j], A_pred[i,j])

np.savetxt('./'+jihe+'/net_dti.txt', A_pred)




