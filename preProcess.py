import csv

#這程式負責資料預處理
#功能是將棋子位置用數字表示
# EX: B[ba] -> B[0100]

data_type = "dan" # dan / kyu

df = open(f'train\{data_type}_train.csv').read().splitlines()

games = [i.split(',',0)[-1] for i in df]
chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}

with open(f'train\{data_type}_train_num.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for game in games:
        for char in chars:
            if (len(str(coordinates[char])) == 1):
                num = '0' + str(coordinates[char])
            else:
                num = str(coordinates[char])
            game = game.replace(char, num)
        gameList = game.split(',')
        writer.writerow(gameList)
print('done')        
