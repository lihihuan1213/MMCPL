import torch
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, input_dim):  # input_dim:300
        super(SiameseNet, self).__init__()
        # Define a fully connected layer to process the input
        self.fc_layer = nn.Linear(input_dim, 64)
        # ReLu activate function
        self.relu_layer = nn.ReLU()
        # Classification layer
        self.classifier_layer = nn.Linear(64 * 4, 1)

    def forward(self, x1, x2):
        c1 = self.relu_layer(self.fc_layer(x1))
        c2 = self.relu_layer(self.fc_layer(x2))
        diff = torch.sub(c1, c2)
        multiply = torch.mul(c1, c2)
        v = torch.cat((c1, c2, diff, multiply), dim=1)
        logit = self.classifier_layer(v)
        return logit