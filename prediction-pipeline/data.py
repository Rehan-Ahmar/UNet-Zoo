import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import os

class TestDataset(Dataset):
    __file = []
    __im = []
    im_ht = 0
    im_wd = 0
    dataset_size = 0

    def __init__(self, images_folder, im_size=[128, 128], transform=None):

        self.__file = []
        self.__im = []
        self.__mask = []
        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.transform = transform
        
        for file in os.listdir(images_folder):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.__file.append(os.path.splitext(file)[0])
                self.__im.append(os.path.join(images_folder, file))
        self.dataset_size = len(self.__file)

    def __getitem__(self, index):
        img = Image.open(self.__im[index]).convert('L')
        img = img.resize((self.im_ht, self.im_wd))
        if self.transform is not None:
            img_tr = self.transform(img)
        return img_tr

    def __len__(self):
        return len(self.__im)
