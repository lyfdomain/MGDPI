import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_size, input_size2, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),  # 隐藏层1
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(512, hidden_size),  # 隐藏层2
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_size2, 2048),  # 隐藏层1
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(512, hidden_size),  # 隐藏层2
            nn.Tanh(),
        )
    def forward(self, x, y):
        encoded = self.encoder(x)
        decoded = self.decoder(y)
        return encoded, decoded

class AEFSLoss(nn.Module):
    def __init__(self):
        super().__init__()  
    def forward(self, e, sr, f, sp, dti, cof, dti2):  
        Sd = torch.mm(e, e.T)
        sn = torch.mm(torch.sqrt_(torch.sum(e.mul(e), dim=1).view(torch.sum(e.mul(e), dim=1).shape[0], 1)),
                      torch.sqrt_(torch.sum(e.mul(e), dim=1).view(1, torch.sum(e.mul(e), dim=1).shape[0])))
        SN = torch.div(Sd, sn)
        los1 = torch.sum((SN - sr) ** 2) / (SN.shape[0] ** 2)
        Sp = torch.mm(f, f.T)
        snp = torch.mm(torch.sqrt_(torch.sum(f.mul(f), dim=1).view(torch.sum(f.mul(f), dim=1).shape[0], 1)),
                       torch.sqrt_(torch.sum(f.mul(f), dim=1).view(1, torch.sum(f.mul(f), dim=1).shape[0])))
        SNp = torch.div(Sp, snp)
        los2 = torch.sum((SNp - sp) ** 2) / (SNp.shape[0] ** 2)
        S3 = torch.mm(e, f.T)
        sn3 = torch.mm(torch.sqrt_(torch.sum(e.mul(e), dim=1).view(torch.sum(e.mul(e), dim=1).shape[0], 1)),
                       torch.sqrt_(torch.sum(f.mul(f), dim=1).view(1, torch.sum(f.mul(f), dim=1).shape[0])))
        SN3 = torch.div(S3, sn3)
        los3 = torch.sum((SN3 - dti) ** 2 * cof) / (torch.sum(cof))

        los4=torch.sum((SN3 - dti2) ** 2) / (SN3.shape[0] * SN3.shape[1])
        los = 1* los3 + 2*los1 + 2*los2 + 2*los4
        return los
