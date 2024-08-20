import torch.nn as nn
import torch
import math, copy
import torch.nn.functional as F

dim = 50

def clones(module, N):
    """Product N identical layers."""
    # print("clones!")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("scores size: ", str(scores.size()))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class Attention(nn.Module):
    def __init__(self, h=1, d_model=100, dropout=0.2):
        """Take in model size and number of heads."""
        super(Attention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        # self.initialize_weights()

    def forward(self, x, mask=None):
        query = x
        key = query
        value = query
        tmp = x
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # print('Before transform query: ', str(query.size()))
        # (batch_size, seq_length, d_model)
        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch_size, h, seq_length, d_k)
        # print('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x) + tmp



class OuterProductMean(nn.Module):
    def __init__(self, ic=50, tp=140):
        super(OuterProductMean, self).__init__()
        self.ic = ic
        self.tp = tp
        self.linear1_1 = nn.Linear(self.ic,100)
        self.linear1_2 = nn.Linear(self.ic,100)
        self.linear2 = nn.Linear(100*self.tp, self.ic*self.ic)
        self.bm = nn.BatchNorm1d(self.ic*self.ic)

    def forward(self, x):

        out1 = self.linear1_1(x)
        out2 = self.linear1_2(x)
        out1 = out1.view(out1.size(0), out1.size(1), out1.size(2), -1)
        out2 = out2.view(out2.size(0), out2.size(1), -1, out2.size(2))
        out = torch.multiply(out1, out2)
        out = torch.mean(out, 3)
        out = torch.flatten(out, 1)
        out = self.linear2(out)
        out = self.bm(out)

        return out.view(out.size(0), self.ic, self.ic)

class ResB(nn.Module):
    def __init__(self, inc=1, outc=1, ks=1, pad=0):
        super(ResB, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc,kernel_size=ks,stride=1, padding=pad)
        self.ln = nn.LayerNorm(50)

    def forward(self, x):
        out = self.conv(x)
        out = self.ln(out)
        out += x
        return out

class TC_Pathway(nn.Module):

    def __init__(self):
        super(TC_Pathway, self).__init__()
        # self.RL1 = nn.ReLU()
        self.sAtt1 = Attention(h=1, d_model=50)
        self.tAtt1 = Attention(h=1, d_model=100)
        self.ln1 = nn.LayerNorm(100)
        self.sAtt2 = Attention(h=5, d_model=50)
        self.tAtt2 = Attention(h=5, d_model=100)
        self.ln2 = nn.LayerNorm(50)
        # self.sAtt3 = Attention(h=10, d_model=50)
        # self.tAtt3 = Attention(h=10, d_model=100)
        # self.ln3 = nn.LayerNorm(100)
        self.LR = nn.Linear(50, 2)

    def forward(self, tcs):
        # out = self.RL1(tcs)
        out = self.sAtt1(tcs)
        out = self.tAtt1(out.transpose(1, 2))
        out = self.ln1(out)
        tcmap1 = out

        out = self.tAtt2(out)
        out = self.sAtt2(out.transpose(1, 2))
        out = self.ln2(out)
        tcmap2 = out

        out = out.transpose(1, 2)
        # out = self.sAtt3(out)
        # out = self.tAtt3(out.transpose(1, 2))
        # out = self.ln3(out)
        # tcmap3 = out

        out = torch.mean(out, 2)
        tcbm = out
        out = self.LR(out)

        # return out, tcmap1, tcmap2, tcmap3, tcbm
        return out, tcmap1, tcmap2, tcbm

class Mask(nn.Module):
    def __init__(self, ic=50, tp=50):
        super(Mask, self).__init__()
        self.ic = ic
        self.tp = tp
        self.ap = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.mp = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bn = nn.Tanh()
        self.linear1 = nn.Linear(self.ic, self.ic)
        self.linear2 = nn.Linear(self.ic, self.ic)
    def forward(self, x):
        # out = self.bn(x)
        ap = self.ap(self.linear1(x))
        mp = self.mp(self.linear2(x))
        # print(ap.shape, mp.shape)
        # cosine sim
        csim = nn.functional.cosine_similarity(ap, mp)
        # 将csim扩展到与x相同的维度
        csim = csim.unsqueeze(1)
        # print(csim.shape)
        # csim与x相乘
        out = torch.multiply(csim, x)
        # print(out.shape)
        return out

class my_model(nn.Module):

    def __init__(self, num_classes=2, ic=50, tp=110):
        super(my_model, self).__init__()
        self.ic = ic
        self.tp = tp
        self.num_classes = num_classes
        self.bm1 = nn.BatchNorm1d(self.ic)
        self.bm2 = nn.BatchNorm1d(self.tp)
        self.bm3 = nn.BatchNorm1d(self.ic)
        self.bm4 = nn.BatchNorm1d(self.ic)
        self.bm5 = nn.BatchNorm1d(self.ic)
        self.bm6 = nn.BatchNorm1d(self.tp)

        self.rltc1_1 = nn.SiLU()
        self.sAtt1 = Attention(h=1, d_model=self.ic) # 50, 64
        self.rltc1_2 = nn.SiLU()
        self.tAtt1 = Attention(h=1, d_model=self.tp)
        self.rltc1_3 = nn.SiLU()
        self.fcAtt1 = Attention(h=1, d_model=self.ic)
        self.rl1 = nn.ReLU()
        # self.bm1 = nn.BatchNorm2d()
        # self.ln1 = nn.LayerNorm(100)
        self.rltc2_1 = nn.SiLU()
        self.sAtt2 = Attention(h=1, d_model=self.ic)
        self.rltc2_2 = nn.SiLU()
        self.tAtt2 = Attention(h=1, d_model=self.tp)
        self.rltc2_3 = nn.SiLU()
        self.fcAtt2 = Attention(h=1, d_model=self.ic)
        self.rl2 = nn.SiLU()
        # self.ln2 = nn.LayerNorm(50)

        self.opm1 = OuterProductMean(ic=self.ic, tp=self.tp)
        self.opm2 = OuterProductMean(ic=self.ic, tp=self.tp)
        self.opm3 = OuterProductMean(ic=self.ic, tp=self.tp)

        self.glb = Mask(ic=self.ic, tp=self.tp)

        self.fc = nn.Linear(5*self.ic, self.ic)
        self.bm = nn.BatchNorm1d(self.ic)
        self.relu = nn.SiLU()
        # self.fctail = nn.Linear(50, 50)


        self.LR = nn.Linear(self.ic, self.num_classes)

        self.SM = nn.Softmax(dim=1)
        self.Act = nn.SiLU()
        # self.Act = nn.Tanh()
        # self.Act = nn.Sigmoid()
        # self.Act = nn.SiLU()
        # self.initialize_weights()

        self.f2t1 = nn.Linear(2*self.ic, self.tp)
        self.f2t2 = nn.Linear(2*self.ic, self.tp)
        self.t2f1 = nn.Linear(self.ic, 2*self.ic)
        self.t2f2 = nn.Linear(self.tp, 2*self.ic)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, tcs, fnc, s=1, eva=False):
        tcs = tcs.transpose(1, 2)
        glob = self.glb(fnc)
        globbm = torch.mean(glob, 2)
        fout = self.fcAtt1(fnc)
        fout = self.rl1(fout)
        fout = self.bm1(fout)
        fncmap1 = torch.cat((fout, glob), dim=-1)

        tcout = self.rltc1_1(tcs)
        tcout = self.sAtt1(tcout)
        tcout = self.rltc1_2(tcout)
        tcout = self.bm2(tcout)
        tcout = self.tAtt1(tcout.transpose(1, 2))
        tcout = self.rltc1_3(tcout)
        tcout = self.bm3(tcout)

        tcmap1 = tcout
        tcmap1 = self.opm1(tcmap1.transpose(1, 2))

        tcout = tcout + self.f2t1(fncmap1)
        fout = fout + tcmap1
        fncbm1 = torch.mean(fout, 2)

        fout = self.fcAtt2(fout)
        fout = self.rl2(fout)
        fout = self.bm4(fout)
        fncmap2 = torch.cat((fout, glob), dim=1)

        tcout = self.tAtt2(tcout)
        tcout = self.rltc2_1(tcout)
        tcout = self.bm5(tcout)
        tcout = self.sAtt2(tcout.transpose(1, 2))
        tcout = self.rltc2_2(tcout)
        tcout = self.bm6(tcout)

        tcbm1 = torch.mean(tcout.transpose(1, 2), 2)

        tcout = self.rltc2_3(tcout)
        tcmap2 = tcout
        tcmap2 = self.opm2(tcmap2)

        tcout = tcout.transpose(1, 2)+self.f2t2(fncmap2.transpose(1, 2))
        fout = fout+tcmap2

        tcout = torch.mean(tcout, 2)
        tcbm2 = tcout

        fout = torch.mean(fout, 2)
        fncbm2 = fout

        bm = torch.cat((globbm, tcbm1, fncbm1, tcbm2, fncbm2), dim=1)
        bm = self.Act(bm) # SiLu, Sigmoid
        # bm = self.arcface(bm)

        if eva:
            return bm

        bm = self.fc(bm)
        bm = self.bm(bm)
        bm = self.relu(bm)

        out = self.LR(bm)
        # print("Out",out.shape)
        # return F.normalize(out, dim=-1)*s
        return out, bm

class LogisticRegression(nn.Module):
    def __init__(self, input=2):
        super(LogisticRegression, self).__init__()
        self.input = input
        self.linear = nn.Linear(self.input,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.linear(x))

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = my_model(num_classes=2, ic=50, tp=140)
    net.to(device)

    tcs = torch.randn([10, 50, 140]).to(device)
    fnc = torch.randn((10, 50, 50)).to(device)

    # tcs = torch.randn([10, 50, 100]).to(device)
    # fnc = torch.randn((10, 50, 50)).to(device)

    out, bm = net(tcs, fnc)

    print(bm.shape, out.shape)