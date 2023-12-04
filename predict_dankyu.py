import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}
chartonumbers = {k:v for k,v in enumerate(chars)}
target_file = './ps.csv'

def search(all, target, row, column, visited, live):
    visited.add((row, column))
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, down, left, right
    for dr, dc in directions:
        r, c = row + dr, column + dc
        if 0 <= r <= 18 and 0 <= c <= 18 and (r, c) not in visited:
            if all[r][c] == 0 and target[r][c] == 0:
                return True
            elif target[r][c] == 1:
                live = search(all, target, r, c, visited, live)
                if live:
                    return live
    return False

def dieCheck(all, target, row, column):
    num = 4
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, down, left, right
    for dr, dc in directions:
        r, c = row + dr, column + dc
        if r < 0 or r > 18 or c < 0 or c > 18 or all[r][c] == 1:
            num -= 1
    if num == 0:
        target[row][column] = 0
        return target
    else:
        num -= sum(target[row + dr][column + dc] == 1 for dr, dc in directions if 0 <= row + dr <= 18 and 0 <= column + dc <= 18)
        if num == 0:
            live = False
            visited = set()
            for dr, dc in directions:
                r, c = row + dr, column + dc
                if 0 <= r <= 18 and 0 <= c <= 18 and target[r][c] == 1 and (r, c) not in visited:
                    live = search(all, target, r, c, visited, live)
                    if live:
                        break
            if not live:
                for r, c in visited:
                    target[r][c] = 0
    return target

def twoColor(moves, previous_board=None):
    if previous_board is None:
        blackAll = np.zeros((19,19))
        whiteAll = np.zeros((19,19))
    else:
        blackAll, whiteAll = previous_board
        moves = [moves[-1]]

    for i in moves:
        column = int(i[2:4])
        row = int(i[4:6])
        color = i[0]
        if color == 'B':
            blackAll[row][column] = 1
            target = whiteAll
            all = blackAll
        else:
            whiteAll[row][column] = 1
            target = blackAll
            all = whiteAll

        arr = [0, 0, 0, 0]
        if row > 0 and target[row-1][column] == 1:
            arr[0] = 1
        if row < 18 and target[row+1][column] == 1:
            arr[1] = 1
        if column > 0 and target[row][column-1] == 1:
            arr[2] = 1
        if column < 18 and target[row][column+1] == 1:
            arr[3] = 1

        for i in range(4):
            if arr[i]==1:
                if i==0:
                    target = dieCheck(all, target=target, row=row-1, column=column)
                elif i==1:
                    target = dieCheck(all, target=target, row=row+1, column=column)
                elif i==2:
                    target = dieCheck(all, target=target, row=row, column=column-1)
                elif i==3:
                    target = dieCheck(all, target=target, row=row, column=column+1)

        if color == 'B':
            whiteAll = target
        else:
            blackAll = target

    return blackAll, whiteAll

def liberties(board):
    liberties_board = np.zeros((19,19))
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, down, left, right
    visited = set()
    liberties_visited = set()

    for row in range(19):
        for column in range(19):
            if board[row][column] != 0 and (row, column) not in visited:  # If there is a stone at this position
                group = [(row, column)]
                liberties_count = 0
                liberties_visited.clear()
                visited.add((row, column))

                # Find all stones in the same group
                i = 0
                while i < len(group):
                    r, c = group[i]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr <= 18 and 0 <= nc <= 18:
                            if board[nr][nc] == 0 and (nr, nc) not in liberties_visited:  # If the adjacent position is empty
                                liberties_count += 1
                                liberties_visited.add((nr, nc))
                            elif board[nr][nc] == board[row][column] and (nr, nc) not in visited:  # If the adjacent stone is the same color
                                group.append((nr, nc))
                                visited.add((nr, nc))
                    i += 1

                # Assign the liberties count to all stones in the same group
                for r, c in group:
                    liberties_board[r][c] = liberties_count
    return liberties_board

def prepare_input_for_dan_models(moves):

    array = np.zeros((11,19,19))
    
    target_step = len(moves)
    # 前五步
    if target_step > 4:
        array[0], array[1] = twoColor(moves[: target_step - 4])
    # 前四步
    if target_step > 3:
        array[2], array[3] = twoColor([moves[target_step - 4]], previous_board=(array[0], array[1]))
    # 前三步
    if target_step > 2:
        array[4], array[5] = twoColor([moves[target_step - 3]], previous_board=(array[2], array[3]))
    # 前二步
    if target_step > 1:
        array[6], array[7] = twoColor([moves[target_step - 2]], previous_board=(array[4], array[5]))
    # 前一步
    array[8], array[9] = twoColor([moves[target_step - 1]], previous_board=(array[6], array[7]))
    
    # 計算氣
    # array[10] = array[8] + array[9]
    array[10] = liberties(array[8] + array[9] * 2) / 10

    return array.transpose((1, 2, 0))

def prepare_input_for_kyu_models(moves):

    array = np.zeros((11,19,19))
    
    target_step = len(moves)
    # 前五步
    if target_step > 4:
        array[0], array[1] = twoColor(moves[: target_step - 4])
    # 前四步
    if target_step > 3:
        array[2], array[3] = twoColor([moves[target_step - 4]], previous_board=(array[0], array[1]))
    # 前三步
    if target_step > 2:
        array[4], array[5] = twoColor([moves[target_step - 3]], previous_board=(array[2], array[3]))
    # 前二步
    if target_step > 1:
        array[6], array[7] = twoColor([moves[target_step - 2]], previous_board=(array[4], array[5]))
    # 前一步
    array[8], array[9] = twoColor([moves[target_step - 1]], previous_board=(array[6], array[7]))
    
    # 計算氣
    array[10] = array[8] + array[9]
    # array[10] = liberties(array[8] + array[9] * 2) / 10

    return array.transpose((1, 2, 0))


def number_to_char(number):
    number_1, number_2 = divmod(number, 19)
    return chartonumbers[number_1] + chartonumbers[number_2]

def top_5_preds_with_chars(predictions):
    resulting_preds_numbers = []
    for prediction in predictions:
        tmp = []
        for i in range(1, 6):
            tmp.append(np.argpartition(prediction, -i)[-i])  # 取第幾大的index
        resulting_preds_numbers.append(tmp)
    resulting_preds_chars = np.vectorize(number_to_char)(resulting_preds_numbers)
    return resulting_preds_chars

### Dan
import danKyu_model

model = danKyu_model.create_model()
model.load_weights('./dan_best.h5')

# 資料不須經過preProcess.py處理
file_list = ["./dan_test_public.csv", 
             "./dan_test_private.csv"]

def prepare_label(move):
    column = int(move[2:4])
    row = int(move[4:6])
    return column*19+row

for data_set in file_list:
    # Load the corresponding dataset
    df = open(data_set).read().splitlines()
    games_id = [i.split(',',2)[0] for i in df]
    games = [i.split(',',2)[-1] for i in df]

    x_testing = []
    y_testing = []

    for game in games:
        for char in chars:
            if (len(str(coordinates[char])) == 1):
                num = '0' + str(coordinates[char])
            else:
                num = str(coordinates[char])
            game = game.replace(char, num)
        gameList = game.split(',')
        x_testing.append(prepare_input_for_dan_models(gameList))

    x_testing = np.array(x_testing)
    predictions = model.predict(x_testing)
    prediction_chars = top_5_preds_with_chars(predictions)

    with open(target_file,'a') as f:
        for index in range(len(prediction_chars)):
            answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
            f.write(answer_row)


### Kyu
model = danKyu_model.create_model()
model.load_weights('./kyu_best.h5')

# 資料不須經過preProcess.py處理
file_list = ["./kyu_test_public.csv", 
             "./kyu_test_private.csv"]

for data_set in file_list:
    # Load the corresponding dataset
    df = open(data_set).read().splitlines()
    games_id = [i.split(',',2)[0] for i in df]
    games = [i.split(',',2)[-1] for i in df]

    x_testing = []

    for game in games:
        for char in chars:
            if (len(str(coordinates[char])) == 1):
                num = '0' + str(coordinates[char])
            else:
                num = str(coordinates[char])
            game = game.replace(char, num)
        gameList = game.split(',')

        x_testing.append(prepare_input_for_kyu_models(gameList))

    x_testing = np.array(x_testing)
    predictions = model.predict(x_testing)
    prediction_chars = top_5_preds_with_chars(predictions)

    with open(target_file,'a') as f:
        for index in range(len(prediction_chars)):
            answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
            f.write(answer_row)