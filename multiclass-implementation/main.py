# %% -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torchvision.transforms as tr

from data import BraTSDatasetUnet, TestDataset, BraTSDatasetLSTM
from losses import DICELossMultiClass
from models import UNet


def train(model, epoch, loss_list, train_loader, optimizer, criterion, args):
    model.train()
    for batch_idx, (image, mask) in enumerate(train_loader):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, mask)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage DICE Loss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def generate_colorimg(output):
    colors = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255]]
    _, height, width = output.shape
    colorimg = np.zeros((height, width, 3), dtype=np.uint8)
    #colorimg = np.full((height, width, 3), 255, dtype=np.uint8)
    #colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    for y in range(height):
        for x in range(width):
            selected_color = colors[output[0,y,x]]
            colorimg[y,x,:] = selected_color
    return tr.ToTensor()(colorimg.astype(np.uint8))

def test(model, loader, criterion, args, validation=False, save_output=False):
    test_loss = 0
    model.eval()
    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        with torch.no_grad():
            image, mask = Variable(image), Variable(mask)
            pred = model(image)
            #pred = torch.sigmoid(pred)
            maxes, out = torch.max(pred, 1, keepdim=True)
        
        if save_output:
            save_image(image, './output/images/images-batch-{}.png'.format(batch_idx))
            save_image(mask, './output/masks/masks-batch-{}.png'.format(batch_idx))
            save_image(out, './output/predictions/outputs-batch-{}.png'.format(batch_idx)) # normalize=True for multiclass
            new_outs = []
            for o in out:
                new_outs.append(generate_colorimg(o))
            save_image(new_outs, './output/final/out-batch-{}.png'.format(batch_idx))
        
            np.save('./npy-files/out-files/{}-batch-{}-outs.npy'.format(args.save, batch_idx),
                    out.data.cpu().numpy())
            np.save('./npy-files/out-files/{}-batch-{}-masks.npy'.format(args.save, batch_idx),
                    mask.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-batch-{}-images.npy'.format(args.save, batch_idx),
                    image.data.byte().cpu().numpy())

        test_loss += criterion(pred, mask).item()
    # Average Dice Coefficient
    test_loss /= len(loader)
    if validation:
        print('\nValidation Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))
    else:
        print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))

def test_only(model, loader, criterion, args):
    model.eval()
    for batch_idx, image in tqdm(enumerate(loader)):
        if args.cuda:
            image = image.cuda()

        with torch.no_grad():
            image = Variable(image)
            pred = model(image)
            #pred = torch.sigmoid(pred)
            maxes, out = torch.max(pred, 1, keepdim=True)
            
        save_image(image, './test-output/images/images-batch-{}.png'.format(batch_idx))
        save_image(out, './test-output/predictions/outputs-batch-{}.png'.format(batch_idx))
        new_outs = []
        for o in out:
            new_outs.append(generate_colorimg(o))
        save_image(new_outs, './test-output/final/out-batch-{}.png'.format(batch_idx))

def start():
    parser = argparse.ArgumentParser(description='UNet + BDCLSTM for BraTS Dataset')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N', help='input batch size for testing (default: 4)')
    parser.add_argument('--train', action='store_true', default=False, help='Argument to train model (default: False)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training (default: False)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='batches to wait before logging training status')
    parser.add_argument('--size', type=int, default=128, metavar='N', help='imsize')
    parser.add_argument('--load', type=str, default=None, metavar='str', help='weight file to load (default: None)')
    parser.add_argument('--data', type=str, default='./Data/', metavar='str', help='folder that contains data')
    parser.add_argument('--save', type=str, default='OutMasks', metavar='str', help='Identifier to save npy arrays with')
    parser.add_argument('--modality', type=str, default='flair', metavar='str', help='Modality to use for training (default: flair)')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='str', help='Optimizer (default: SGD)')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    DATA_FOLDER = args.data

    # %% Loading in the model
    # Binary
    # model = UNet(num_channels=1, num_classes=2)
    # Multiclass
    model = UNet(num_channels=1, num_classes=3)

    if args.cuda:
        model.cuda()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99)
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Defining Loss Function
    criterion = DICELossMultiClass()

    if args.train:
        # %% Loading in the Dataset
        full_dataset = BraTSDatasetUnet(DATA_FOLDER, im_size=[args.size, args.size], transform=tr.ToTensor())
        #dset_test = BraTSDatasetUnet(DATA_FOLDER, train=False, 
        # keywords=[args.modality], im_size=[args.size,args.size], transform=tr.ToTensor())
    
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        validation_loader = DataLoader(validation_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
        #test_loader = DataLoader(full_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

        print("Training Data : ", len(train_loader.dataset))
        print("Validaion Data : ", len(validation_loader.dataset))
        #print("Test Data : ", len(test_loader.dataset))
    
        loss_list = []
        start = timer()
        for i in tqdm(range(args.epochs)):
            train(model, i, loss_list, train_loader, optimizer, criterion, args)
            test(model, validation_loader, criterion, args, validation=True)
        end = timer()
        print("Training completed in {:0.2f}s".format(end - start))

        plt.plot(loss_list)
        plt.title("UNet bs={}, ep={}, lr={}".format(args.batch_size, args.epochs, args.lr))
        plt.xlabel("Number of iterations")
        plt.ylabel("Average DICE loss per batch")
        plt.savefig("./plots/{}-UNet_Loss_bs={}_ep={}_lr={}.png".format(args.save, args.batch_size, args.epochs, args.lr))

        np.save('./npy-files/loss-files/{}-UNet_Loss_bs={}_ep={}_lr={}.npy'.format(args.save, args.batch_size, args.epochs, args.lr),
                np.asarray(loss_list))
        print("Testing Validation")
        test(model, validation_loader, criterion, args, save_output=True)
        torch.save(model.state_dict(), 'unet-multiclass-model-{}-{}-{}'.format(args.batch_size, args.epochs, args.lr))
        
        print("Testing PDF images")
        test_dataset = TestDataset('./pdf_data/', im_size=[args.size, args.size], transform=tr.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
        print("Test Data : ", len(test_loader.dataset))
        test_only(model, test_loader, criterion, args)
    
    elif args.load is not None:
        test_dataset = TestDataset(DATA_FOLDER, im_size=[args.size, args.size], transform=tr.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
        print("Test Data : ", len(test_loader.dataset))
        model.load_state_dict(torch.load(args.load))
        test_only(model, test_loader, criterion, args)
        #test(model, train_loader, test_loader, criterion, args, save_output=True, train_accuracy=True)

if __name__ == '__main__':
    start()
