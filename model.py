import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z


class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class D2(nn.Module):
    def __init__(self, opt):
        super(D2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.mapping = nn.Linear(opt.resSize + opt.attSize, 4096)
        self.mapping2 = nn.Linear(4096, opt.resSize)
        self.discrim = nn.Linear(2048, 1)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        m = self.lrelu(self.mapping(h))
        m = self.lrelu(self.mapping2(m))
        disc = self.discrim(m)
        return m, disc

class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h

class RNet1(nn.Module):
    def __init__(self, opt):
        super(RNet1, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self,  x_g):
        h = self.lrelu(self.fc1(x_g))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

