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

