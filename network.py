import torch.nn as nn
import torch.nn.functional as F
from torch import tanh


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, outputs)

    def forward(self, x):
        x = tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
