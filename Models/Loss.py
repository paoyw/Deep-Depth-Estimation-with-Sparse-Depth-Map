import torch.nn as nn

class MaskedMSE(nn.MSELoss):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
    
    def forward(self, input, target, masked):
        return super().forward(input[masked], target[masked])