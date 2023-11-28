def render_gamestate(gamestate):
    x = gamestate[0:9]
    o = gamestate[9:]
    tokens = []
    for i in range(9):
        if x[i] == 1:
            tokens.append('X')
        elif o[i] == 1:
            tokens.append('O')
        else:
            tokens.append('_')

    b = f'_{tokens[0]}_|_{tokens[1]}_|_{tokens[2]}_\n' \
        f'_{tokens[3]}_|_{tokens[4]}_|_{tokens[5]}_\n' \
        f'_{tokens[6]}_|_{tokens[7]}_|_{tokens[8]}_\n'

    return b

def mp_game(AI):
    gamestate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    x = gamestate[0:9]
    o = gamestate[9:]
    while evaluate_game(gamestate) == 0:
        x_move = AI.choose_move(x+o)
        x_new = []
        for i in range(9):
            if i == x_move:
                x_new.append(1)
            else:
                x_new.append(o[i])
        x = x_new
        print(render_gamestate(x+o))
        o_move = input('Enter #: ')
        o_new = []
        for i in range(9):
            if i == o_move:
                o_new.append(1)
            else:
                o_new.append(o[i])
        o = o_new
        print(render_gamestate(x + o))
        gamestate = x + o


def evaluate_side(gamestate):
    result = 0
    for i in [0,3,6]:
        if gamestate[i] == gamestate[i+1] == gamestate[i+2] == 1:
            result = 1
    for i in [0,1,2]:
        if gamestate[i] == gamestate[i+3] == gamestate[i+6] == 1:
            result = 1
    if gamestate[0] == gamestate[4] == gamestate[8] == 1 or gamestate[2] == gamestate[4] == gamestate[7] == 1:
        result = 1

    return result

def evaluate_game(gamestate):
    if sum(gamestate) == 9:
        return 0.5
    else:
        my_value =  evaluate_side(gamestate[0:9])
        opponent_value = evaluate_side(gamestate[9:]) * -1
    if my_value !=0:
        return my_value
    if opponent_value !=0:
        return opponent_value
    else:
        return 0

