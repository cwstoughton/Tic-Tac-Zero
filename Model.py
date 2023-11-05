import torch

class TicTacZero(torch.nn.Module):
    def __init__(self):
        super(TicTacZero, self).__init__()
        self.input = torch.nn.Linear(18,18)
        self.dense1 = torch.nn.Linear(18,18)
        self.dense2  = torch.nn.Linear(18,18)
        self.dense3 = torch.nn.Linear(18,18)
        self.output = torch.nn.Linear(18,9)

        self.output_activation = torch.nn.Softmax(9)
        self.inner_activation = torch.nn.ReLU()

    def forward(self,x):
        x = self.input(x)
        x = self.inner_activation(x)
        x = self.dese1(self.inner_activation(x))
        x = self.dese3(self.inner_activation(x))
        x = self.output(self.output_activation(x))

        return x

