import torch
from torch import nn
from torch._C import device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, actionSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.fc1 = nn.Linear(3136, 512)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        # Output 2 values: fly up and do nothing
        self.fc2_pol = nn.Linear(512, self.actionSpaceSize)
        self.fc2_val = nn.Linear(512, 1)
        torch.nn.init.xavier_uniform_(self.fc2_pol.weight)
        torch.nn.init.kaiming_uniform_(self.fc2_val.weight)
        self.relu = nn.ReLU(inplace=True)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x_pol = self.fc2_pol(x)
        x_val = self.fc2_val(x)
        return self.logsoftmax(x_pol), x_val
    