import torch

import cv2


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None, classes=None, target="casual"):
        self.transform = transform
        self.image_path = image_path
        self.data_num = len(image_path)
        self.classes = classes
        self.target = target

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.resize(image, (240, 320))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if self.transform:
            out_image = self.transform(image)
        else:
            out_image = image[idx]
            
        label = self.classes.index(self.target)

        return out_image, label
    
    

class MyDatasetPerImage(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None, classes=None, target="casual"):
        self.transform = transform
        self.image_path = image_path
        self.data_num = len(image_path)
        self.classes = classes
        self.target = target

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.resize(image, (240, 320))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if self.transform:
            out_image = self.transform(image)
        else:
            out_image = image[idx]
            
        label = self.classes.index(self.target)

        return out_image, label