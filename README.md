# 利用權重調整人臉識別系統中的代表特徵向量以提高識別正確率 
### Refining the representative embedding vectors in face recognition systems by weights adjustment to improve the recognition accuracy

### 說明
做論文研究時所使用的程式碼，由於原先程式碼存在著一些~~歷史遺留~~問題可能會導致部分計算功能緩慢，目前正在緩慢重寫當中  
論文連結：[利用權重調整人臉識別系統中的代表特徵向量以提高識別正確率](https://hdl.handle.net/11296/k6bvdn)

### 內容大綱
本實驗使用了Labeled Faces in the Wild(LFW)、PubFig Dataset與Racial Faces in-the-Wild(RFW)資料集並配合MTCNN人臉偵測進行人臉圖像預處理。  
透過Facenet模型獲取人臉特徵向量，人臉特徵向量經由歐式距離與內積向量的計算之後，希望能透過調整距離閥值與權重代表向量尋找最佳參數。  

實驗結果顯示在透過調整距離閥值與權重代表向量之後皆有顯著的提升。  
* 在LFW_5photo資料集中可達到98.94±0.43%的正確率
* 在LFW_3photo資料集中可達到97.93±0.77%的正確率  
* 在Pubfig資料集中可達到96.53±1.22%的正確率
* 在RFW_African資料集中可達到89.05±1.34%的正確率
* RFW_Asian資料集中可達到87.54±1.3%的正確率
* RFW_Indian資料集中可達到92.27±1.16%的正確率
* RFW_Caucasian可達到93.87±1.04%的正確率。

### 程式路徑相關說明
請按照以下路徑創建資料夾：  
![image](https://imgur.com/9wWcTbY.jpg)    
![image](https://imgur.com/QYMmJzg.jpg)  


### 目前功能
[人臉圖片預處理 image_preprocessing](https://github.com/kerycheng/face-recognition-with-embedding-vector-mean/blob/main/image_preprocessing.py)  
使用MTCNN將LFW做人臉預處理，並將處理好的人臉資料集另存至lfw_crops  

[建立資料庫與測試庫 create_dataset_and_testset](https://github.com/kerycheng/face-recognition-with-embedding-vector-mean/blob/main/create_dataset_and_testset.py)  
將lfw_crops按照需求去建立資料庫(dataset)與測試庫(testset)  

[建立dataframe create_dataframe](https://github.com/kerycheng/face-recognition-with-embedding-vector-mean/blob/main/create_dataframe.py)  
計算並創建有關資料庫的dataframe並存放在data/dataset.json  
方便之後直接使用表上的資料進行計算工作  

[基本設定 basic_settings](https://github.com/kerycheng/face-recognition-with-embedding-vector-mean/blob/main/basic_settings.py)  
宣告目錄，建立facenet、MTCNN模型。計算方法相關  


### 參考來源
 * https://github.com/erhwenkuo/face-recognition
