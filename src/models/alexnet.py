import torch.nn as nn
import torch.nn.functional as F
import torch

class fasion_mnist_alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))
        out = self.conv2(out)
        # print(out)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = self.conv3(out)
        # print(out)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = self.conv4(out)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = self.conv5(out)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = F.dropout(out, 0.5)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = F.relu(self.fc2(out))
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = F.dropout(out, 0.5)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = self.fc3(out)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        out = F.log_softmax(out, dim=1)
        # print(len(torch.nonzero(out.view(-1))) / len(out.view(-1)))

        return out