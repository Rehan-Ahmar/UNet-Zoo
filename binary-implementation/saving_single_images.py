
import argparse
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from PIL import Image
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torchvision.transforms as tr

from data import TestDataset
from losses import DICELossMultiClass
from models import UNet

def test_only(model, loader, criterion, sizes, file_names, cuda=False):
    model.eval()
    for batch_idx, image in tqdm(enumerate(loader)):
        if cuda:
            image = image.cuda()
        with torch.no_grad():
            image = Variable(image)
            pred = model(image)
            maxes, out = torch.max(pred, 1, keepdim=True)

        for i, output in enumerate(out):
            back_img = tr.ToPILImage()(output.type(torch.float32))  
            index = batch_idx*4 + i
            back_img = back_img.resize(sizes[index])
            back_img.save(os.path.join('./output/single/', file_names[index]))
        save_image(image, './output/images/images-batch-{}.png'.format(batch_idx))
        save_image(out, './output/predictions/outputs-batch-{}.png'.format(batch_idx))

def start():
    model = UNet(num_channels=1, num_classes=2)
    criterion = DICELossMultiClass()

    dir_path = './Data/images/'
    sizes = []
    file_names = []
    for f in os.listdir(dir_path):
        im = Image.open(os.path.join(dir_path, f))
        print(im.size)
        sizes.append(im.size)
        file_names.append(f)
    print(sizes)
    print(file_names)

    test_dataset = TestDataset('./Data/', im_size=[256, 256], transform=tr.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)
    print("Test Data : ", len(test_loader.dataset))
    model.load_state_dict(torch.load('unet-model-16-100-0.001', map_location='cpu'))
    test_only(model, test_loader, criterion, sizes, file_names)

if __name__ == '__main__':
    start()
