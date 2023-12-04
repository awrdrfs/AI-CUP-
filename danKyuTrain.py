# 有吃子判斷(有稍微優化算法)，沒有算氣，往前看三步 => 6個19*19
# 嘗試使用data generator
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import danKyu_model as modelAndLayer
import random
import matplotlib.pyplot as plt

data_dankyu = "dan"  # dan / kyu

def prepare_label(move):
    column = int(move[2:4])
    row = int(move[4:6])
    return column*19+row

def search(all, target, row, column, visited, live):
    visited.add((row, column))
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
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
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
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

        # 遞迴尋找棋子還有沒有氣
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

# 遞迴計算每個棋子的氣
def liberties(board):
    liberties_board = np.zeros((19,19))
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = set()
    liberties_visited = set()

    for row in range(19):
        for column in range(19):
            if board[row][column] != 0 and (row, column) not in visited:
                group = [(row, column)]
                liberties_count = 0
                liberties_visited.clear()
                visited.add((row, column))

                # 把同組的棋子都找出來
                i = 0
                while i < len(group):
                    r, c = group[i]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr <= 18 and 0 <= nc <= 18:
                            if board[nr][nc] == 0 and (nr, nc) not in liberties_visited:
                                liberties_count += 1
                                liberties_visited.add((nr, nc))
                            elif board[nr][nc] == board[row][column] and (nr, nc) not in visited:  # 沒看過且同顏色的棋子
                                group.append((nr, nc))
                                visited.add((nr, nc))
                    i += 1

                # 同組的棋子都有同樣的氣
                for r, c in group:
                    liberties_board[r][c] = liberties_count
    return liberties_board

# 建立N筆隨機棋局
def dataMaker(gameCount, games, color):
    x = np.zeros((gameCount, 11, 19, 19))
    y = np.zeros((gameCount))
    # 產生隨機的局
    random_game_idx = random.sample(range(0, len(games)), gameCount)
    for idx in range(gameCount):
        # 取出局的index
        i = random_game_idx[idx]
        game = games[i]
        # 根據逗號切開資料，將每一棋存到list裡
        moves_list = game.split(',')
        # 在這局中隨機選一步作為預測目標
        target_step = random.randint(1, len(moves_list) - 1)
        # 如果這步不是指定的顏色，就繼續隨機選
        while moves_list[target_step][0] != color[i]:
            target_step = random.randint(1, len(moves_list) - 1)

        # 將這局的資料存到x，y裡
        y[idx] = prepare_label(moves_list[target_step])
        
        array = np.zeros((11,19,19))
        # 前五步
        if target_step > 4:
            array[0], array[1] = twoColor(moves_list[: target_step - 4])
        # 前四步
        if target_step > 3:
            array[2], array[3] = twoColor([moves_list[target_step - 4]], previous_board=(array[0], array[1]))
        # 前三步
        if target_step > 2:
            array[4], array[5] = twoColor([moves_list[target_step - 3]], previous_board=(array[2], array[3]))
        # 前二步
        if target_step > 1:
            array[6], array[7] = twoColor([moves_list[target_step - 2]], previous_board=(array[4], array[5]))
        
        # 前一步
        array[8], array[9] = twoColor([moves_list[target_step - 1]], previous_board=(array[6], array[7]))
        
        # 計算氣
        if data_dankyu == "dan":
            array[10] = liberties(array[8] + array[9] * 2) / 10  # Dan
        else:
            array[10] = array[8] + array[9]  # Kyu

        x[idx] = array

    return x, y

df = open(f'./train/{data_dankyu}_train_num.csv').read().splitlines()

games = [i.split(',',2)[-1] for i in df]
color = [i.split(',',3)[1] for i in df]

games = np.array(games)
color = np.array(color)

games_train, games_val, color_train, color_val = train_test_split(games, color, test_size=0.05, shuffle=True)

del(df)
del(games)
del(color)

# get validation data (五千局中的五千步)
val_size = 5000
X_test, y_test = dataMaker(val_size, games_val, color_val)
X_test = X_test.transpose((0, 2, 3, 1))
y_test = tf.one_hot(y_test, depth=19*19)
del(games_val)
del(color_val)

model = modelAndLayer.create_model()
model.summary()

# 最小learning_rate，希望learning_rate最多下降10次
miniLR = 0.001 / 1024

# 紀錄最好的ValAccuracy
bestValAccuracy = 0.0

# 紀錄上一次的ValAccuracy
best_val_acc = 0.0
early_stop_count = 0
accuracyUptime = 0

# 設定optimizer，learning rate=0.001
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

epochs = 150
batch_count = 1000
batch_size = 512

# 建立dataset
def data_generator():
    while True:
        yield dataMaker(batch_size, games_train, color_train)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 11, 19, 19), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# 預先載入資料
dataset = dataset.prefetch(20)

for epoch in range(epochs):
    print(f'-----------epoch {epoch+1}  (LR: {float(opt.learning_rate):e})-----------')

    for i, (X_train, y_train) in enumerate(dataset.take(batch_count)):
        X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])
        y_train = tf.one_hot(y_train, depth=19*19)

        # 開始訓練
        with tf.GradientTape() as tape:
            y_pred = model(X_train)
            # Compute loss.
            loss = tf.keras.losses.categorical_crossentropy(y_train, y_pred)
            trainable_variables = model.trainable_variables
            # 計算梯度
            gradients = tape.gradient(loss, trainable_variables)
        opt.apply_gradients(zip(gradients, trainable_variables))

        # 印loss和accuracy
        if (i+1) % (batch_count // 4) == 0:
            print(f'batch {i+1} loss: {sum(loss) / len(loss)}')
            acc = tf.keras.metrics.categorical_accuracy(y_train, y_pred)
            print(f'batch {i+1} accuracy: {sum(acc) / len(acc)}')


    # 驗證模型
    history["train_loss"].append(sum(loss) / len(loss))
    history["train_acc"].append(sum(acc) / len(acc))
    total_val_loss = 0
    total_val_acc = 0
    val_batch_count = 10
    val_batch_size = val_size // val_batch_count
    for i in range(val_batch_count):
        y_pred = model(X_test[ val_batch_size * i : val_batch_size * (i+1) ])
        val_loss = tf.keras.losses.categorical_crossentropy(y_test[ val_batch_size * i : val_batch_size * (i+1) ], y_pred)
        val_loss = sum(val_loss) / len(val_loss)
        total_val_loss += val_loss
        val_acc = tf.keras.metrics.categorical_accuracy(y_test[ val_batch_size * i : val_batch_size * (i+1) ], y_pred)
        val_acc = sum(val_acc) / len(val_acc)
        total_val_acc += val_acc
    
    total_val_loss /= val_batch_count
    total_val_acc /= val_batch_count
    print(f'val loss: {total_val_loss}')
    print(f'val accuracy: {total_val_acc}')
    history["val_loss"].append(total_val_loss)
    history["val_acc"].append(total_val_acc)

    # draw curve
    plt.plot(history["train_acc"])
    plt.plot(history["val_acc"])
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'])
    plt.savefig(f'./{data_dankyu}_accuracy.png')
    plt.clf()
    # draw curve
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'])
    plt.savefig(f'./{data_dankyu}_loss.png')
    plt.clf()

    model.save_weights(f'./{data_dankyu}_last.h5')

    # val_accuracy比上一次好就存檔
    if (bestValAccuracy < total_val_acc):
        bestValAccuracy = total_val_acc
        model.save_weights(f'./{data_dankyu}_best.h5')
        print('weight saved')

    if ((epoch + 1) % 5 == 0):
        model.save_weights(f'./{data_dankyu}_epoch{epoch+1}.h5')
        print('weight saved')
    
    # 連續三次val_accuracy沒有上漲，learning_rate就乘1/2
    if(best_val_acc <= total_val_acc):
        accuracyUptime = 0
        early_stop_count = 0
    else:
        accuracyUptime +=1
        early_stop_count += 1
        if (accuracyUptime == 3 and opt.learning_rate > miniLR):
            opt.learning_rate = (opt.learning_rate)/2
            print('learning rate reduced to ', opt.learning_rate)
            # 下次learning rate下降是五(提高條件)
            accuracyUptime = -2
        # 連續七次val_accuracy沒有上漲，就停止訓練
        if early_stop_count == 7:
            print("-------Early stop-------")
            break

    # 更新 best_val_acc
    if total_val_acc > best_val_acc:
        best_val_acc = total_val_acc
