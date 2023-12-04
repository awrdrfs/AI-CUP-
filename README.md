# 配置環境  
棋力模仿：  
Python 3.9.18  
Tensorflow : 2.10.0  
Keras : 2.10.0  
Scikit-learn : 1.3.2  
numpy : 1.26.1  
matplotlib : 3.8.2  
棋風辨識(colab)：  
Python 3.10.12  
Tensorflow : 2.14.0  
Keras : 2.14.0  
Scikit-learn : 1.2.2  
numpy : 1.23.5  
matplotlib : 3.7.1  
opencv-python : 4.8.0.76  
(不建議使用不同版本，可能會導致程式出錯)  
# 重要模塊輸入/輸出 
dieCheck, twoColor, liberties 棋風辨識、棋力模仿皆有使用
dataMaker 僅為棋力模仿使用
dieCheck(all, target, row, column)  
輸入：all=吃子該顏色的局面、target=被吃子該顏色的局面、row, cloumn=檢查目標的座標  
輸出：若有吃子回傳提子後的 target，若無回傳 target  
twoColor(moves, previous_board=None)  
輸入：moves=previous_board接下來的棋路、previous_board=先前的棋局(預設None)  
輸出：blackAll=黑子在棋盤的分布、whiteAll=白子在棋盤的分布  
liberties(board)  
輸入：board=目標棋盤  
輸出：目標棋盤氣的分布  
dataMaker(gameCount, games, color)  
輸入：gameCount=指定步數、games=資料集、color=指定顏色(黑/白)  
輸出：x=訓練資料、y=驗證資料  
# 棋風辨識說明  
playStyleTrain 用於訓練模型、playStylePredict 用於預測，兩程式都是.ipynb檔可以直接上傳 colab  
playStyleTrain 執行前請確認資料集的路徑，執行時會顯示模型結構與訓練過程，並自動儲存模型權重  
playStylePredict  執行前請確認資料集、權重及預測結果的路徑  
# 棋力模仿說明  
在訓練棋力模仿前，請先使用 preProcess.py 將 dan/kyu 原本的資料從英文座標轉為數字座標，並於 danKyuTrain.py 的11行指定要訓練 dan/kyu  
danKyu_model 是模型架構，在 train,predict 會被 import  
執行 danKyuTrain.py 會顯示模型結構與訓練過程，並自動儲存模型權重  
執行 predict_danKyu.py 會將 dan,kyu 一起預測各自的 public,private ，結果存在同一個.csv檔裡  
