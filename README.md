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
# 棋風辨識說明  
playStyleTrain用於訓練模型、playStylePredict用於預測，兩程式都是.ipynb檔可以直接上傳colab  
# 棋力模仿說明  
在訓練棋力模仿前，請先使用preProcess.py將dan/kyu原本的資料從英文座標轉為數字座標，並於danKyuTrain.py的11行指定要訓練dan/kyu  
執行danKyuTrain.py會顯示模型結構與訓練過程，並自動儲存模型權重  
