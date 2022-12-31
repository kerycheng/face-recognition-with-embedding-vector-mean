import os
import random
import shutil

from basic_settings import setup_settings

# 建立資料庫（dataset）與測試庫（testset)
class dataset_and_testset_preprocessing(setup_settings):
    def __init__(self, faces_number, people_number, seed=10):
        super().__init__()

        self.faces_number = faces_number # 人臉張數
        self.people_number = people_number # 人數
        self.seed = seed # 種子碼

        random.seed(self.seed)

    def run(self):
        self.directory_path()
        self.check_dataset()
        self.create_testset()
        self.random_pick_testphoto()

        statistical(self.dataset_dir, os.listdir(self.testset_dir))

    # 查看人臉資料集的人臉數量與人數的比例
    def check_dataset(self):
        self.crops_path = os.path.join(self.root_dir, self.img_out_path)
        self.crops_dir = os.listdir(self.crops_path)

        faces_count = {} # 計數字典
        count_number = [] # 人臉張數存進陣列裡

        for count in range(len(self.crops_dir)):
            photo = os.listdir(os.path.join(self.crops_path, self.crops_dir[count]))
            count_number.append(len(photo))

        for number in count_number:
            if number in faces_count:
                faces_count[number] += 1
            else:
                faces_count[number] = 1
        print(dict(sorted(faces_count.items())))

    # 將大於等於某數人臉張數的人放入dataset資料夾
    def create_testset(self):
        dataset = os.path.join(self.root_dir, self.dataset_path) # 要挑選並儲存的資料集
        if not os.path.exists(dataset): # 如果沒有資料集就建一個
            os.makedirs(dataset)

        count = 0
        for i in range(len(self.crops_dir)):
            photo = os.listdir(os.path.join(self.crops_path, self.crops_dir[i]))

            if len(photo) == self.faces_number:
                copy_dir = os.path.join(self.crops_path, self.crops_dir[i])
                paste_dir = os.path.join(dataset, self.crops_dir[i])
                print(copy_dir, paste_dir)
                shutil.copytree(copy_dir, paste_dir)
                count += 1

            if count == self.people_number:
                break

    # 每個人隨機抽出一張照片放入測試集
    def random_pick_testphoto(self):
        self.dataset_dir = os.listdir(self.dataset_path)

        self.testset_dir = os.path.join(self.root_dir, self.testdata_path)
        if not os.path.exists(self.testset_dir): # 如果沒有資料集就建一個
            os.makedirs(self.testset_dir)

        for i in range(len(self.dataset_dir)):
            photo = os.listdir(os.path.join(self.dataset_path, self.dataset_dir[i]))
            testing_number = random.randint(0, len(photo) - 1)
            testing_photo = photo[testing_number]

            testing_file_path = os.path.join(self.dataset_path, self.dataset_dir[i])
            move_testing_photo = os.path.join(testing_file_path, testing_photo)

            shutil.move(move_testing_photo, self.testset_dir)

        count = 0
        for i in range(len(self.crops_dir)):
            photo = os.listdir(os.path.join(self.crops_path, self.crops_dir[i]))
            if len(photo) != self.faces_number:
                testing_number = random.randint(0, len(photo) - 1)
                testing_photo = photo[testing_number]

                testing_file_path = os.path.join(self.crops_path, self.crops_dir[i])
                move_testing_photo = os.path.join(testing_file_path, testing_photo)

                shutil.move(move_testing_photo, self.testset_dir)
                count += 1

            if count == self.people_number:
                break


class statistical(object):
    def __init__(self, dataset, testset):
        self.dataset = dataset
        self.testset = testset

        self.statistical_testset()

    # 統計測試集
    def statistical_testset(self):
        inface = 0

        for i in range(len(self.dataset)):
            for j in range(len(self.testset)):
                if self.dataset[i] in self.testset[j]:
                    inface += 1

        print(self.dataset)
        print(self.testset)

        print(f'在庫人臉: {inface}')
        print(f'非在庫人臉: {len(self.testset) - inface}')

if __name__ == '__main__':
    dtp = dataset_and_testset_preprocessing(4, 100) # (臉的張數, 人數)
    dtp.run()