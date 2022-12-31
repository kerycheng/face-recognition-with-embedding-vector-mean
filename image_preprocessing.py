# 屏蔽Jupyter的warning訊息
import warnings
warnings.filterwarnings('ignore')

# Utilities相關函式庫
from scipy import misc
import os
from tqdm import tqdm

# 多維向量處理相關函式庫
import numpy as np

# 專案相關函式庫
import facenet
import detect_face

from basic_settings import setup_settings

class images_prepeocessing(setup_settings):
    def __init__(self):
        super().__init__()

    def run(self):
        self.directory_path()
        self.check_directory()
        self.build_mtcnn()
        self.mtcnn_parameter()
        self.build_random_key()
        self.face_detect_clip()

    # 檢查output資料夾，若無則建立新的並打印出該資料夾有多少人
    def check_directory(self):
        if not os.path.exists(self.img_out_path):
            os.makedirs(self.img_out_path)

        self.dataset = facenet.get_dataset(self.img_in_path)
        # 打印看有多少人
        print(f"Total face identities: {len(self.dataset)}")

    def build_random_key(self):
        # 將一個隨機key添加到圖像檔名以允許使用多個進程進行人臉對齊
        random_key = np.random.randint(0, high=99999)
        self.bounding_boxes_filename = os.path.join(self.img_out_path, 'bounding_boxes_%05d.txt' % random_key)

    def face_detect_clip(self):
        # 使用Tensorflow來運行MTCNN
        with open(self.bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0  # 處理過的圖像總數
            nrof_successfully_aligned = 0  # 人臉圖像align的總數

            # 迭代每一個人臉身份(ImageClass)
            for cls in tqdm(self.dataset):
                output_class_dir = os.path.join(self.img_out_path, cls.name)  # 裁剪後的圖像目錄
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)

                # 迭代每一個人臉身份的圖像的路徑 (ImageClass.image_paths)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]  # 取得圖像檔名
                    output_filename = os.path.join(output_class_dir, filename + '.png')  # 設定輸出的圖像檔名
                    # print(image_path)

                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)  # 讀進圖檔
                            # print('read data dimension: ', img.ndim)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            # print(errorMessage)
                        else:
                            # 將圖檔轉換成numpy array (height, widith, color_channels)
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)
                            img = img[:, :, 0:3]
                            # print('after data dimension: ', img.ndim)

                            # 使用MTCNN來偵測人臉在圖像中的位置
                            bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold,
                                                                        self.factor)
                            nrof_faces = bounding_boxes.shape[0]  # 偵測到的人臉總數
                            # print('detected_face: %d' % nrof_faces)
                            if nrof_faces > 0:
                                # 當有偵測到多個人臉的時候, 我們希望從中找到主要置中位置的人臉
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det = det[index, :]
                                det = np.squeeze(det)
                                bb_temp = np.zeros(4, dtype=np.int32)
                                # 取得人臉的左上角與右下角座標
                                bb_temp[0] = det[0]
                                bb_temp[1] = det[1]
                                bb_temp[2] = det[2]
                                bb_temp[3] = det[3]

                                # 進行裁剪以及大小的轉換
                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                scaled_temp = misc.imresize(cropped_temp, (self.image_size, self.image_size), interp='bilinear')

                                nrof_successfully_aligned += 1
                                misc.imsave(output_filename, scaled_temp)  # 儲存處理過的圖像
                                text_file.write('%s %d %d %d %d\n' % (
                                output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                            else:
                                # print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

if __name__ == '__main__':
    ip = images_prepeocessing()
    ip.run()