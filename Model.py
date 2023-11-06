import torch
import numpy as np
class TicTacZero(torch.nn.Module):
    def __init__(self):
        super(TicTacZero, self).__init__()
        self.input = torch.nn.Linear(18,18)
        self.dense1 = torch.nn.Linear(18,18)
        self.dense2 = torch.nn.Linear(18,18)
        self.dense3 = torch.nn.Linear(18,18)
        self.output = torch.nn.Linear(18,9)

        self.output_activation = torch.nn.Softmax(9)
        self.inner_activation = torch.nn.ReLU()

        #loss = sum(value@boardstate_i - game_outcome)^2
        #self.optimizer = torch.optim.SGD()

    def _preprocess_boardstate(self, boardstate):
        return torch.FloatTensor(boardstate)
    def forward(self,x):
        x = self._preprocess_boardstate(x)
        x = self.input(x)
        x = self.dense1(self.inner_activation(x))
        x = self.dense2(self.inner_activation(x))
        x = self.output(x)

        return x

    def choose_move(self, boardstate):
        x = np.array(boardstate[0:9])
        o = np.array(boardstate[9:])
        legal_moves = (x + o - 1) * -1
        legal_moves = torch.FloatTensor(legal_moves)
        move_values = self.forward(boardstate)
        legal_move_values = move_values * legal_moves
        move_choice = [0+int(i == max(legal_move_values)) for i in legal_move_values]
        return move_choice

    def play_game(self):
        boardstate = [0,0,0,0,0,0,0,0,0,
                      0,0,0,0,0,0,0,0,0]




