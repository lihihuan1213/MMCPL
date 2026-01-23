import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., bias=True):
        # Initialize the graph convolution layer
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))    # Weight parameter
        # Optional bias parameter
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()    # Initialize weights and bias

    def reset_parameters(self):
        # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Forward pass: input features are transformed using the weight matrix
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        # String representation of the layer
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate=0.5):
        super(GCN, self).__init__()
        # Define graph convolution layers
        self.gc1 = GraphConvolution(nfeat, nhid)  # 768,512 
        # self.gc1 = GraphConvolution(nfeat, nclass) # 
        self.gc2 = GraphConvolution(nhid, nclass)  # 512,256
        # self.gc3 = GraphConvolution(nclass, nclass)

        # Define fully connected layers to map input features to hidden and output layers
        self.linear1 = nn.Linear(nfeat, nhid)
        # self.linear1 = nn.Linear(nfeat, nclass)
        self.linear2 = nn.Linear(nhid, nclass)
        # self.linear3 = nn.Linear(nclass, nclass)

        # Define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        # First graph convolution + ReLU activation + Fully connected layer + Dropout
        x1 = self.dropout(F.relu(self.gc1(x, adj)) + self.linear1(x))
        # Second graph convolution + ReLU activation + Fully connected layer + Dropout
        x2 = self.dropout(F.relu(self.gc2(x1, adj)) + self.linear2(x1))
        # If a third graph convolution layer is needed, apply similarly
        # x3 = self.dropout(F.relu(self.gc3(x2, adj)) + self.linear3(x2))
        return x2
