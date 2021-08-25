import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        x = x.view(x.size(0), -1).contiguous()
        out = self.logistic(x)
        # out = F.log_softmax(out, dim=1)
        return out


def LROnMnist():
    return LogisticRegression(28 * 28, 10)
