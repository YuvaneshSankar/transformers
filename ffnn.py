import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self,X,dropout_rate=0.1):
        super(FFNN,self).__init__()
        self.fl1 = nn.Linear(X, 4 * X)  # creates bias as well
        self.fl2 = nn.Linear(4 * X, X)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self,x):
        x=self.fl1(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.fl2(x)
        return x
