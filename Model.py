import torch
import numpy as np
import game
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

        self.game_memory = []

        #loss = sum(value@boardstate_i - game_outcome)^2
        #self.optimizer = torch.optim.SGD()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _preprocess_boardstate(self, boardstate):
        return torch.FloatTensor(boardstate)
    def forward(self,x):
        x = self._preprocess_boardstate(x)
        x = self.input(x)
        x = self.dense1(self.inner_activation(x))
        x = self.dense2(self.inner_activation(x))
        x = self.dense3(self.inner_activation(x))
        x = self.output(self.inner_activation(x))
      #  x = self.inner_activation(x)
     #   x = torch.softmax(x,0)
        return x

    def choose_move(self, boardstate):
        x = np.array(boardstate[0:9])
        o = np.array(boardstate[9:])
        legal_moves = (x + o - 1) * -1
        legal_moves = torch.FloatTensor(legal_moves)
        move_values = self.forward(boardstate)
        legal_move_values = (move_values * legal_moves) + legal_moves
        move_choice = [0+int(i == torch.argmax(legal_move_values)) for i in range(9)]
        return move_choice

    def reset_memory(self):
        self.game_memory = []

    def play_game(self):
        boardstate = [0,0,0,0,0,0,0,0,0,
                      0,0,0,0,0,0,0,0,0]

        game_status = 0
        memory = []
        #print('starting loop')
        while game_status == 0:
            x = boardstate[0:9]
            o = boardstate[9:]
            if sum(o) == sum(x):
                move = self.choose_move(boardstate)
                x = [sum(i) for i in zip(x, move)]
                token = 'x'
            else:
                move = self.choose_move(boardstate)
                o = [sum(i) for i in zip(o, move)]
                token = 'o'

            boardstate = x+o
            memory.append(boardstate)
            game_status = game.evaluate_game(boardstate, token)

        return memory, game_status

    def learn(self):
        x,y = self.play_game()
        y = [y for i in range(9)]
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.4)
        loss_fn = torch.nn.L1Loss()
        for i in x:
            optimizer.zero_grad()
            y_hat = self.choose_move(i)
            loss = loss_fn(torch.FloatTensor(y_hat),torch.FloatTensor(y))
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
    def fit(self, y):
        x = self.game_memory
        y = [y for y in range(9)]
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.4)
        loss_fn = torch.nn.L1Loss()
        for i in x:
            optimizer.zero_grad()
            y_hat = self.choose_move(i)
            loss = loss_fn(torch.FloatTensor(y_hat),torch.FloatTensor(y))
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
        self.reset_memory()



class TrainingEnvironment:
    def __init__(self):
        self.player_1 = TicTacZero()
        self.player_2 = TicTacZero()
        self.x_wins = 0
        self.o_wins = 0
        self.ties = 0

    def reset_log(self):
        self.x_wins = 0
        self.o_wins = 0
        self.ties = 0

    def run_game(self):
        x = [0,0,0,0,0,0,0,0,0]
        o = [0,0,0,0,0,0,0,0,0]
        game_status = 0
        while game_status == 0:
            x_move = self.player_1.choose_move(x+o)
            x = [sum(i) for i in zip(x, x_move)]
            self.player_1.game_memory.append(x+o)
            game_status = game.evaluate_game(x+o)
            if game_status == 0:
                o_move = self.player_2.choose_move(o+x)
                o = [sum(i) for i in zip(o, o_move)]
                self.player_2.game_memory.append(x + o)
                game_status = game.evaluate_game(o+x)
        x_score = game.evaluate_game(x+o)
        o_score = game.evaluate_game(o+x)
        self.player_1.fit(x_score)
        self.player_2.fit(o_score)
        if x_score > o_score:
            self.x_wins += 1
        elif x_score < o_score:
            self.o_wins += 1
        else:
            self.ties += 1

    def epoch(self,n_games):
        for i in range(n_games):
            self.run_game()
        if self.x_wins/n_games > 0.6:
            self.player_2.load_state_dict(self.player_1.state_dict())
        elif self.o_wins/n_games > 0.6:
            self.player_1.load_state_dict(self.player_2.state_dict())
        self.reset_log()





def epoch(n_games):

    player_1 = TicTacZero()
    player_2 = TicTacZero()
    def play_game(self):
        boardstate = [0,0,0,0,0,0,0,0,0,
                      0,0,0,0,0,0,0,0,0]

        game_status = 0
        memory = []
        #print('starting loop')
        while game_status == 0:
            x = boardstate[0:9]
            o = boardstate[9:]
            if sum(o) == sum(x):
                move = self.choose_move(boardstate)
                x = [sum(i) for i in zip(x, move)]
                token = 'x'
            else:
                move = self.choose_move(boardstate)
                o = [sum(i) for i in zip(o, move)]
                token = 'o'

            boardstate = x+o
            memory.append(boardstate)
            game_status = game.evaluate_game(boardstate, token)

        return memory, game_status

