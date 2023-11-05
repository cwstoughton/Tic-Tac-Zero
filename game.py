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

def evaluate_game(gamestate, token):
    if sum(gamestate) == 9:
        return 0.1
    else:
        if token == 'x':
            return evaluate_side(gamestate[0:9])
        elif token == 'o':
            return evaluate_side(gamestate[9:])

