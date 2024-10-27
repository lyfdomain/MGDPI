
import copy
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from tqdm import trange
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
# from Bio.SubsMat import MatrixInfo as matlist

drug_data=[]
jihe="dataset/biosnap"
prenum=2181

with open("./"+jihe+"/full.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        drug_data.append(row)
drug_data= np.array(drug_data)
print(drug_data[0])


drug=[]
for i in drug_data[:,0]:
    if i not in drug:
        drug.append(i)

protein=[]
for j in drug_data[:,1]:
    if j not in protein:
        protein.append(j)

print(len(drug), len(protein))
np.savetxt("./"+jihe+"/drug.txt",  np.array(drug), fmt="%s" )
np.savetxt("./"+jihe+"/protein.txt", np.array(protein), fmt="%s" )

dti=np.zeros((len(drug),len(protein)))
for i in range(len(drug_data)):
    if drug_data[i,0] in drug and drug_data[i,1] in protein:
        dti[drug.index(drug_data[i,0]) ,protein.index(drug_data[i,1])] = int(float(drug_data[i,2]))
np.savetxt("./"+jihe+"/dti.txt", dti )

drug_data_train=[]
with open("./"+jihe+"/train.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        drug_data_train.append(row)
drug_data_train= np.array(drug_data_train)

dti_train=np.zeros((len(drug),len(protein)))
for i in range(len(drug_data_train)):
    if drug_data_train[i,0] in drug and drug_data_train[i,1] in protein:
        dti_train[drug.index(drug_data_train[i,0]) ,protein.index(drug_data_train[i,1])] = int(float(drug_data_train[i,2]))
np.savetxt("./"+jihe+"/dti_train.txt", dti_train )
print(dti_train.shape)

drug_data_val=[]
with open("./"+jihe+"/val.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        drug_data_val.append(row)
drug_data_val= np.array(drug_data_val)

index_val=[]
for i in range(len(drug_data_val)):
    if drug_data_val[i,0] in drug and drug_data_val[i,1] in protein:
        index_val.append(drug.index(drug_data_val[i,0])*prenum+protein.index(drug_data_val[i,1]))
np.savetxt("./"+jihe+"/index_val.txt", np.array(index_val))

drug_data_test=[]
with open("./"+jihe+"/test.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        drug_data_test.append(row)
drug_data_test= np.array(drug_data_test)

index_test=[]
for i in range(len(drug_data_test)):
    if drug_data_test[i,0] in drug and drug_data_test[i,1] in protein:
        index_test.append(drug.index(drug_data_test[i,0])*prenum+protein.index(drug_data_test[i,1]))
np.savetxt("./"+jihe+"/index_test.txt", np.array(index_test))