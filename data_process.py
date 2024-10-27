import numpy as np
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


with open("./"+jihe+"/full.csv", newline='') as csvfile:
    # 使用csv模块读取CSV文件
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    # 遍历CSV文件中的每一行数据
    for row in reader:
        # 处理每一行数据
        # print(', '.join(row))
        drug_data.append(row)
# drug_data=np.loadtxt("./unseen_drug/train.csv", delimiter=',',dtype=str, skiprows=1)
drug_data= np.array(drug_data)
print(drug_data[0])

# print(drug_data_p)
drug=[]
for i in drug_data[:,0]:
    if i not in drug:
        drug.append(i)

protein=[]
for j in drug_data[:,1]:
    if j not in protein:
        protein.append(j)


np.savetxt("./"+jihe+"/drug.txt",  np.array(drug), fmt="%s" )
np.savetxt("./"+jihe+"/protein.txt", np.array(protein), fmt="%s" )

dti=np.zeros((len(drug),len(protein)))
for i in range(len(drug_data)):
    if drug_data[i,0] in drug and drug_data[i,1] in protein:
        dti[drug.index(drug_data[i,0]) ,protein.index(drug_data[i,1])] = int(float(drug_data[i,2]))
np.savetxt("./"+jihe+"/dti.txt", dti )
print(dti.shape)

set1 = copy.deepcopy(drug)
set2 = copy.deepcopy(drug)

# 将SMILES字符串转换为分子对象列表
mol_list1 = [Chem.MolFromSmiles(smiles) for smiles in set1]
mol_list2 = [Chem.MolFromSmiles(smiles) for smiles in set2]

# 计算分子指纹
fp_list1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_list1]
fp_list2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_list2]
# print(fp_list2)
# 计算相似性矩阵
similarity_matrix = np.zeros((len(fp_list1), len(fp_list2)))
for i in trange(len(fp_list1)):
    for j in range(len(fp_list2)):
        similarity = DataStructs.DiceSimilarity(fp_list1[i], fp_list2[j])
        similarity_matrix[i, j] = similarity

print(similarity_matrix,similarity_matrix.shape)
np.savetxt("./"+jihe+"/DS.txt", similarity_matrix )


# 定义两个蛋白质集合
records1 = copy.deepcopy(protein)
records2  = copy.deepcopy(protein)

# 定义史密斯-沃特曼比对器
aligner = PairwiseAligner()
aligner.mode = 'global'

aligner.open_gap_score = -5
aligner.extend_gap_score = -1
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")


score_matrix = np.zeros((len(records1), len(records2)))
for i in trange(len(records1)):
    for j in range(len(records2)):
        # alignments = Bio.Align.PairwiseAligner.align.globalds(records1[i], records2[j], matrix, gap_open, gap_extend)
        # print(str(records2[j]))
        char_to_remove = "U"
        records1[i] = records1[i].replace(char_to_remove, "")
        records2[j] = records2[j].replace(char_to_remove, "")
        alignments = aligner.align(str(records1[i]), str(records2[j])).score
        # score = max([alignment.score for alignment in alignments])[0]
        score_matrix[i, j] = alignments


print(score_matrix)
np.savetxt("./"+jihe+"/PS.txt",score_matrix)


