from image_preprocessing import images_prepeocessing
from create_dataset_and_testset import dataset_and_testset_preprocessing

if __name__ == '__main__':
    ip = images_prepeocessing()
    ip.run()

    dtp = dataset_and_testset_preprocessing(4, 100)
    dtp.run()