from torch import nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 color channels, 6 out, 5 kernel
        self.pool = nn.MaxPool2d(2, 2) # kernel 2, stride 2
        self.conv2 = nn.Conv2d(6, 16, 5) # input = last out
        self.fc1 = nn.Linear(16*47*47, 120) # 16ch, 120 hidden
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7) # 7 classes
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*47*47) # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # no softmax since the loss contains it
        return x