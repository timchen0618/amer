import torch.nn as nn

# Define model
class LinearProbe(nn.Module):
    def __init__(self):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(768, 11)  # 12 classes

    def forward(self, _input):
        logits = self.linear(_input)
        return logits

# Define linear regression model
class LinearRegressionProbe(nn.Module):
    def __init__(self):
        super(LinearRegressionProbe, self).__init__()
        self.linear = nn.Linear(768, 1)  # Single output for regression

    def forward(self, _input):
        output = self.linear(_input)
        return output
    
class MLPProbe(nn.Module):
    def __init__(self):
        super(MLPProbe, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 11)
        )

    def forward(self, _input):
        logits = self.layers(_input)
        return logits

class MLPRegressionProbe(nn.Module):
    def __init__(self):
        super(MLPRegressionProbe, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 1)
        )

    def forward(self, _input):
        output = self.layers(_input)
        return output