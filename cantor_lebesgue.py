import torch
from torch import nn

class Cantor(nn.Module):

    def __init__(self, precision):
        super(Cantor, self).__init__()
        self.precision = int(precision)
        self.largeconst = torch.FloatTensor([2**13]).cuda()

        exps = torch.arange((precision - 1), -1, -1)
        self.powersof2 = 2**exps
        self.powersof2 = self.powersof2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.multiplier = torch.FloatTensor([2 ** -precision]).cuda()

        self.powersof2 = self.powersof2.float().cuda()
        print(self.precision, self.largeconst)

    def forward(self, x):
        mw = self.largeconst * x
        numerator = mw - torch.floor(mw)
        gpath = numerator / self.largeconst

        with torch.no_grad():
            cx = x.detach()
            mask = torch.ones_like(cx)
            output = torch.zeros_like(cx).unsqueeze(0).repeat(self.precision, 1, 1, 1)
            og_ones = torch.floor(cx)
            newx = cx - og_ones

            for i in range(0, self.precision):
                temp = newx * 3
                left = torch.floor(temp)
                right = temp - left
                left = left * mask
                lones = left.eq(1.0).float()
                mask = mask - lones
                output[i] = left
                newx = right

            output = output.clamp_max(1)
            output = output.mul(self.powersof2)
            output = output.sum(dim=0)
            output = output.mul(self.multiplier)
            output = output.add(og_ones)
            output = output.sub(gpath.detach())
        return output + gpath