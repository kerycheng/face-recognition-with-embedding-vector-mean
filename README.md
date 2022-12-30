# 利用權重調整人臉識別系統中的代表特徵向量以提高識別正確率 
### Refining the representative embedding vectors in face recognition systems by weights adjustment to improve the recognition accuracy

### 內容大綱
探討使用facenet所獲得的人臉特徵向量，當把同一人的特徵向量相加之後對於人臉辨識是否會有更好的表現

### 程式路徑相關說明
請按照以下路徑創建資料夾：  
![image](https://imgur.com/n5UbRWh.jpg)  

### 目前功能
image_preprocessing.py -> 使用MTCNN將LFW做人臉預處理，並將處理好的人臉資料集另存至lfw_crops  
create_dataset_and_testset.py -> 將lfw_crops按照需求去建立資料庫(dataset)與測試庫(testset)  

### 參考來源
 * https://github.com/erhwenkuo/face-recognition
