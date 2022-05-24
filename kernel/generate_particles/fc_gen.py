import torch
import torch.nn as nn
from kernel.args import get_args
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FCGenerator(nn.Module):
    args = get_args()
    def __init__(self,init_var=0.1,num_p=args.num_particles):
        super(FCGenerator,self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               uniform_(x, -init_var / 2, init_var / 2), nn.init.calculate_gain('relu'))
        self.fc = init_(nn.Linear(1,num_p*3))
        self.input = torch.ones((1,1)).cuda()
    def forward(self):
        x = self.fc(self.input)
        x = x.view(x.size(0),-1)

        return x

