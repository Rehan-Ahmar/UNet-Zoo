# %% -*- coding: utf-8 -*-

from PIL import Image
import os
from timeit import default_timer as timer
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torchvision.transforms as tr

from data import TestDataset
from models import UNet
from draw_bbox import generate_bbox

def predict(model, loader, batch_size, sizes, file_names, masks_path, use_gpu):
    model.eval()
    for batch_idx, image in tqdm(enumerate(loader)):
        if use_gpu:
            image = image.cuda()
        with torch.no_grad():
            image = Variable(image)
            pred = model(image)
            maxes, out = torch.max(pred, 1, keepdim=True)

        for i, output in enumerate(out):
            back_img = tr.ToPILImage()(output.type(torch.float32))  
            index = batch_idx*batch_size + i
            back_img = back_img.resize(sizes[index])
            back_img.save(os.path.join(masks_path, file_names[index]))

def run_pipeline(root_dir, model_path, img_size, batch_size, use_gpu):
    images_path = os.path.join(root_dir, "images")
    masks_path = os.path.join(root_dir, "masks")
    outputs_path = os.path.join(root_dir, "outputs")
    
    sizes = []
    file_names = []
    for f in os.listdir(images_path):
        im = Image.open(os.path.join(images_path, f))
        sizes.append(im.size)
        file_names.append(f)
        
    model = UNet(num_channels=1, num_classes=2)
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    test_dataset = TestDataset(images_path, im_size=[img_size, img_size], transform=tr.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    print("Test Data : ", len(test_loader.dataset))
    start = timer()
    predict(model, test_loader, batch_size, sizes, file_names, masks_path, use_gpu)
    end = timer()
    print("Prediction completed in {:0.2f}s".format(end - start))
    generate_bbox(images_path, masks_path, outputs_path)
    end2 = timer()
    print("Bbox generation completed in {:0.2f}s".format(end2 - end))

if __name__ == '__main__':
    run_pipeline(root_dir='./Data/', model_path='unet-model-binary-dilated-16-100-0.001', img_size=256, batch_size=16, use_gpu=False)
