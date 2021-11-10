import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
import glob
import random
from PIL import Image



class WearDataset(Dataset):

    def __init__(self, subset_indices, classes, data_source_dir=None, image_preprocessing=None):

        self.image_path_list = []
        self.classes = classes

        class_dir_list = [data_source_dir + f"/{class_name}" for class_name in self.classes]
        for class_dir in class_dir_list:
            class_images_path_list = glob.glob(class_dir + "/*.jpg")
            for img_number in subset_indices:
                self.image_path_list.append(class_images_path_list[img_number])

        self.image_preprocessing = image_preprocessing



    def __len__(self):
        return len(self.image_path_list)



    #データセットの読み込み処理の流れ
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]

        #画像読み込み
        image = Image.open(image_path).convert("RGB")

        #読み込んだ画像にdata augmentationを実行する
        image = data_augmentation(image)

        #前処理によってtorch.tensorに変換
        image = self.image_preprocessing(image)

        #画像ファイルパスからクラス名だけを抜き出す
        class_name = image_path.split("/")[-2]

        #抜き出したクラス名をclassesと照合してindexをラベル値とする
        label = self.classes.index(class_name)

        return image, label, image_path.split("/")[-2] + image_path.split("/")[-1]


#学習用画像のData Augmentation
def data_augmentation(input_image):

    #resize
    input_image = input_image.resize((240, 320))

    #random horizontal flip
    if random.random() < 0.5:
        input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)

    #please add your favorite augmentations...

    return input_image


