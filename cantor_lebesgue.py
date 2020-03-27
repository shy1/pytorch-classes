import torch
from torch import nn


class Cantor(nn.Module):
    """
    Cantor-Lebesgue By Proxy
    - expects float32 input between 0 and 1
    - Cantor function part detached from gradient backpropagation and ignored by autograd

    - gradient_path part contributes only tiny value to output but always gives the input a derivative of 1, allowing
    the non-continuously differentiable cantor-lebesgue 'devil's staircase' function to be used as an
    activation function with a 'proxy' gradient for SGD

    - as the function maps 0 --> 0, 1/2 --> 1/2, and 1 --> 1, using a constant linear derivative of 1 seems okay as that
    is sort of? the mean derivative of the function even if it is actually 0 everywhere the actual cantor function is
    differentiable

    """

    def __init__(self, precision):
        super(Cantor, self).__init__()
        self.precision = int(precision)
        self.largeconst = torch.FloatTensor([2**13]).cuda()

        # array of exponents counting down from (precision - 1) to 0
        exps = torch.arange((precision - 1), -1, -1)

        # ..., 2^3, 2^2, 2^1, 2^0
        self.powersof2 = 2**exps
        self.powersof2 = self.powersof2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 2 raised to the value of negative 'precision', ex: 2^-15 = 1/32768
        self.pointmover = torch.FloatTensor([2 ** -precision]).cuda()

        self.powersof2 = self.powersof2.float().cuda()
        print(self.precision, self.largeconst)

    def forward(self, x):
        mw = self.largeconst * x
        numerator = mw - torch.floor(mw)
        gradient_path = numerator / self.largeconst

        with torch.no_grad():
            cx = x.detach()
            mask = torch.ones_like(cx)
            output = torch.zeros_like(cx).unsqueeze(0).repeat(self.precision, 1, 1, 1)

            # convert any 1.00's to 0.00 to make math work using only places to the right of decimal point
            # output in this case will be 0.00 and the 1. is added back at the end to map 1.00 --> 1.00
            og_ones = torch.floor(cx)
            newx = cx - og_ones

            # convert base10 to base3 representation, set all values of 1 to 0 except for the first 1 encountered
            for i in range(0, self.precision):
                # can multiply by 3 since value is always between 0 and 1.0
                temp = newx * 3
                left = torch.floor(temp)
                right = temp - left

                # set 'bit' value to 0 if a preceding decimal place was a 1
                left = left * mask

                # set mask used above if the current decimal place is a 1
                lones = left.eq(1.0).float()
                mask = mask - lones

                output[i] = left
                newx = right

            # convert all 2's in the ternary base3 repr. to 1's
            output = output.clamp_max(1)

            # interpret array of 1's and 0's created above as a base2 binary integer
            output = output.mul(self.powersof2)
            output = output.sum(dim=0)

            # move decimal place back to correct position to create float
            output = output.mul(self.pointmover)

            # add back the 1.0's we took out earlier
            output = output.add(og_ones)

            # subtract tiny gradient_path value to balance out when it is added in return statement so result is same as
            # it would be without adding the gradient path. source paper didn't do this, not sure if for a good reason
            output = output.sub(gradient_path.detach())
        return output + gradient_path