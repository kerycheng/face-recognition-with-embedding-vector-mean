import itertools
import os

import cv2
import scipy
import tensorflow as tf
import detect_face
import numpy as np

import facenet


class setup_settings(object):
    def directory_path(self):
        # 專案的根目錄路徑
        self.root_dir = os.getcwd()
        # 訓練/驗證用的資料目錄
        self.data_path = os.path.join(self.root_dir, "data")
        # 資料集的目錄
        self.dataset_path = os.path.join(self.root_dir, "dataset")
        # 測試集的目錄
        self.testdata_path = os.path.join(self.root_dir, "testset")
        # 模型的資料目錄
        self.model_path = os.path.join(self.root_dir, "model")
        # MTCNN的模型
        self.mtcnn_model_path = os.path.join(self.model_path, "mtcnn")
        # Facenet的模型
        self.facenet_model_path = os.path.join(self.model_path, 'facenet', '20170512-110547', '20170512-110547.pb')
        # 訓練/驗證用的圖像資料目錄
        self.img_in_path = os.path.join(self.data_path, "lfw")
        # 訓練/驗證用的圖像資料目錄
        self.img_out_path = os.path.join(self.data_path, "lfw_crops")

    # 建立MTCNN模型
    def build_mtcnn(self):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, self.mtcnn_model_path)

    # 設定MTCNN的參數
    def mtcnn_parameter(self):
        self.minsize = 20  # 最小的臉部的大小
        self.threshold = [0.4, 0.6, 0.6]  # 三個網絡(P-Net, R-Net, O-Net)的閥值
        self.factor = 0.709  # scale factor

        self.margin = 33  # 在裁剪人臉時的邊框margin
        self.image_size = 182  # 160 + 22

    def build_facenet(self):  # 設置facenet
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.tf_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.tf_sess.as_default()

        facenet.load_model(self.facenet_model_path)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

class calculation(setup_settings):
    def __init__(self):
        super().__init__()

        self.directory_path()
        self.build_facenet()

    # L2歸一化
    def l2_normalize(self, emb, axis=None, epsilon=1e-12):
        return np.divide(emb, np.sqrt(np.maximum(np.sum(np.square(emb)), epsilon)))

    # 計算歐式距離
    def euclidean_distance(self, emb1, emb2):  # 計算歐式距離
        return np.sqrt(np.sum(np.square(emb1 - emb2)))

    # 計算內積向量
    def innerproduct(self, emb1, emb2):
        return np.dot(emb1, emb2)

    # 取得單張圖片特徵向量
    def get_embs(self, img):  # 計算圖片的特徵向量
        image_size = 160
        scaled_reshape = []
        image = scipy.misc.imread(img, mode='RGB')
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        image = facenet.prewhiten(image)
        scaled_reshape.append(image.reshape(-1, image_size, image_size, 3))
        emb = np.zeros((1, self.embedding_size))
        emb = self.tf_sess.run(self.embeddings, feed_dict={self.images_placeholder: scaled_reshape[0], self.phase_train_placeholder: False})[0]

        return emb

    # 取得目錄所有圖片特徵向量
    def get_dir_all_embs(self): # 計算目錄底下的特徵向量, 回傳人名(dataset_dir)、特徵向量陣列(embs_list)、人臉照片陣列(images_name_list)
        self.dataset_dir = os.listdir(self.dataset_path)  # 讀目錄名稱
        self.embs_list = []
        self.images_name_list = []

        for i in range(len(self.dataset_dir)):
            dataset_images_name = os.path.join(self.dataset_path, str(self.dataset_dir[i]))  # 讀目錄底下圖片名稱
            dataset_images_path = sorted(
                [os.path.join(dataset_images_name, f) for f in os.listdir(dataset_images_name)])  # 目錄路徑 + 圖片名稱之後做排序
            self.images_name_list.append(sorted([f for f in os.listdir(dataset_images_name)])) # 將每張圖片名稱放入images_name_list
            self.embs_list.append([self.get_embs(img) for img in dataset_images_path])  # 計算embs放入embs_list

    # 取得每個人的代表向量
    def get_vector_mean(self):
        self.vector_mean_list = [self.l2_normalize(np.divide(sum(np.array(self.embs_list[i])), len(self.embs_list[i]))) for i in range(len(self.embs_list))]

    # 自身特徵向量與自身特徵向量比對
    def self_emb2emb_list(self):
        self.eu_self_emb2emb = []
        self.ip_self_emb2emb = []

        for embs in self.embs_list:
            eul = [self.euclidean_distance(embs[i], embs[j]) for (i, j) in itertools.combinations(range(len(embs)), 2)]
            ipl = [self.innerproduct(embs[i], embs[j]) for (i, j) in itertools.combinations(range(len(embs)), 2)]
            self.eu_self_emb2emb.append((min(eul), max(eul), np.std(eul, ddof=1), np.mean(eul)))
            self.ip_self_emb2emb.append((min(ipl), max(ipl), np.std(ipl, ddof=1), np.mean(ipl)))
            # 最短距離、最遠距離、標準差距離、平均距離

    # 自身特徵向量與他人特徵向量比對
    def other_emb2emb_list(self):
        self.eu_other_emb2emb = []
        self.ip_other_emb2emb = []
        eu_self_emb2emb_min = [self.eu_self_emb2emb[i][0] for i in range(len(self.eu_self_emb2emb))]
        ip_self_emb2emb_min = [self.ip_self_emb2emb[i][0] for i in range(len(self.ip_self_emb2emb))]

        for i in range(len(self.embs_list)):
            # eoed = eu_other_emb2emb_distance
            # ioed = ip_other_emb2emb_distance
            eoed = []
            ioed = []

            for j in range(len(self.embs_list[i])):
                eul = min([self.euclidean_distance(self.embs_list[i][n], self.embs_list[j][m])
                           for (n, m) in itertools.combinations(range(len(self.embs_list[i])), 2)])
                ipl = min([self.innerproduct(self.embs_list[i][n], self.embs_list[j][m])
                           for (n, m) in itertools.combinations(range(len(self.embs_list[i])), 2)])
                # 把min當作該人的代表向量
                if eul == eu_self_emb2emb_min[i]:  # 如果 ddl == 自己的 min_ed 則 pass
                    pass
                else:
                    eoed.append(eul)

                if ipl == ip_self_emb2emb_min[i]:  # 如果 ddl == 自己的 min_ed 則 pass
                    pass
                else:
                    ioed.append(ipl)

            self.eu_other_emb2emb.append((min(eoed), max(eoed), np.std(eoed, ddof=1), np.mean(eoed)))
            self.ip_other_emb2emb.append((min(ioed), max(ioed), np.std(ioed, ddof=1), np.mean(ioed)))

    # 自身代表向量(vector mean)與自身特徵向量比對
    def self_vm2emb_list(self):
        self.eu_self_vm2emb = []
        self.ip_self_vm2emb = []

        for i in range(len(self.vector_mean_list)):
            # eu_self_vm2emb_distance
            # ip_self_vm2emb_distance
            esvd = []
            isvd = []

            for j in range(len(self.embs_list[i])):
                eul = [self.euclidean_distance(self.vector_mean_list[i], self.embs_list[i][j])]
                ipl = [self.innerproduct(self.vector_mean_list[i], self.embs_list[i][j])]
                esvd.append(eul)
                isvd.append(ipl)
            self.eu_self_vm2emb.append(((min(esvd)[0]), (max(esvd)[0]), np.std(esvd, ddof=1), np.mean(esvd)))
            self.ip_self_vm2emb.append(((min(isvd)[0]), (max(isvd)[0]), np.std(isvd, ddof=1), np.mean(isvd)))

    # 自身代表向量禹他人特徵向量比對
    def other_vm2emb_list(self):
        self.eu_other_vm2emb = []
        self.ip_other_vm2emb = []
        eu_self_vector_mean_min = [(self.eu_self_vm2emb[i][0]) for i in range(len(self.eu_self_vm2emb))]
        ip_self_vector_mean_min = [(self.ip_self_vm2emb[i][0]) for i in range(len(self.ip_self_vm2emb))]

        for i in range(len(self.vector_mean_list)):
            # eu_other_vm2emb_distance
            # ip_other_vm2emb_distance
            eovd = []
            iovd = []
            for j in range(len(self.embs_list)):
                eul = min([self.euclidean_distance(self.vector_mean_list[i], self.embs_list[j][m]) for m in range(len(self.embs_list[i]))])
                ipl = min([self.innerproduct(self.vector_mean_list[i], self.embs_list[j][m]) for m in range(len(self.embs_list[i]))])

                if eul == eu_self_vector_mean_min[i]:
                    pass
                else:
                    eovd.append(eul)

                if ipl == ip_self_vector_mean_min[i]:
                    pass
                else:
                    iovd.append(ipl)

            self.eu_other_vm2emb.append((min(eovd), max(eovd), np.mean(eovd), np.std(eovd, ddof=1)))
            self.ip_other_vm2emb.append((min(iovd), max(iovd), np.mean(iovd), np.std(iovd, ddof=1)))