#!/usr/bin/env python
# coding: utf-8
# First, import the package
import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import os
import math
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import pandas as pd

from datetime import datetime


def read_split_data(root: str, val_rate: float = 0.4):
    random.seed(0)  # Ensure the reproducibility of random results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Traverse folders, one folder corresponds to one category
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort to ensure consistent order across all platforms
    flower_class.sort()
    # Generate category names and corresponding numerical indexes
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    #Define different folder names and their corresponding tag numbers
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all image paths of the training set
    train_images_label = []  # Store the corresponding index information of the training set images
    val_images_path = []  # Store all image paths for the validation set
    val_images_label = []  # Store the index information corresponding to the validation set images
    every_class_num = []  # Store the total number of samples for each category
    # Supported file suffix types
    supported = [".NPY", ".npy"]  
    # Traverse the files in each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # Traverse to obtain all supported file paths
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sort to ensure consistent order across all platforms
        images.sort()
        # Retrieve the index corresponding to the category
        image_class = class_indices[cla]
        # Record the number of samples in this category
        every_class_num.append(len(images))
        # Randomly sample verification samples in proportion
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # If the path is in the sampled validation set, it will be stored in the validation set
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # Otherwise, store it in the training set
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # Draw a bar chart of the number of each category
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # Replace the horizontal axis 0,1,2,3,4 with the corresponding category name
        plt.xticks(range(len(flower_class)), flower_class)
        # Add numerical labels on the bar chart
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # Set x coordinate
        plt.xlabel('image class')
        # Set y coordinate
        plt.ylabel('number of images')
        # Set the title of the bar chart
        plt.title('flower class distribution')
        plt.show()
    
    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # The Anti-Normalize Operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove the scale on the x-axis
            plt.yticks([])  # Remove the y-axis scale
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated losses
    accu_num = torch.zeros(1).to(device)   # Accumulate the number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Accumulate the number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # Accumulated losses

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class MyDataSet(Dataset):
    """Custom dataset"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)    
    
    def __getitem__(self, item):
        img = np.load(self.images_path[item])  
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img = transform(img) 
        
        img=img.transpose(0,1)
        img=img.transpose(1,2)
        
        """
        if self.transform is not None:
            img = self.transform(img)        
        """        
        
        label = self.images_class[item]
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

# Define the GARN model components
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class GARN_Downsampler(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(GARN_Downsampler, self).__init__(*layers)

class GARB(nn.Module):
    def __init__(self, n_feats, n_factors):
        super(GARB, self).__init__()

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = nn.Conv2d(n_feats, n_feats // n_factors, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(n_feats // n_factors, n_feats, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        w = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))
        w = self.fc1(w)
        w = self.relu2(w)
        w = self.fc2(w)
        w = torch.sigmoid(w)

        out = out * w
        out = out + x
        return out

class GARN(nn.Module):
    def __init__(self, in_channels, num_blocks=4, n_feats=32, n_factors=4, conv=default_conv):
        super(GARN, self).__init__()

        kernel_size = 3
        out_channels = in_channels // 2

        self.head = nn.Sequential(conv(in_channels, n_feats, kernel_size))
        body_layers = [GARB(n_feats, n_factors) for _ in range(num_blocks)]
        body_layers.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body_layers)

        self.tail = nn.Sequential(
            GARN_Downsampler(n_feats, out_channels, kernel_size),
            conv(out_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# Define the CAF model components
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, near_band, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2 * near_band
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask = None):
        # Unfold input images into patches
        b, c, h, w = x.shape
        patch_size = int(np.sqrt(self.patch_to_embedding.in_features // c))
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(b, c, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(b, -1, patch_size * patch_size * c)

        # Patch embedding
        x = self.patch_to_embedding(patches)
        b, n, _ = x.shape

        # Add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer processing
        x = self.transformer(x, mask)

        # Classification using the cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        return self.mlp_head(x)

# Define the combined GARN_CAF model
class GARN_CAF(nn.Module):
    def __init__(self, garn_in_channels, garn_num_blocks, garn_n_feats, garn_n_factors, caf_params):
        super(GARN_CAF, self).__init__()

        # Initialize GARN model
        self.garn = GARN(
            in_channels=garn_in_channels,
            num_blocks=garn_num_blocks,
            n_feats=garn_n_feats,
            n_factors=garn_n_factors
        )

        # Initialize CAF model with dynamic parameters based on GARN output
        self.caf = ViT(
            image_size=caf_params['image_size'], 
            patch_size=caf_params['patch_size'], 
            near_band=garn_in_channels // 2,  # GARN outputs channels // 2
            num_classes=caf_params['num_classes'], 
            dim=caf_params['dim'], 
            depth=caf_params['depth'], 
            heads=caf_params['heads'], 
            mlp_dim=caf_params['mlp_dim'], 
            dim_head=caf_params['dim_head'], 
            dropout=caf_params['dropout'], 
            emb_dropout=caf_params['emb_dropout'], 
            mode=caf_params['mode']
        )

    def forward(self, x):
        # Pass input through GARN
        garn_output = self.garn.forward(x)

        # Pass GARN output through CAF
        caf_output = self.caf.forward(garn_output)

        return caf_output


    



import argparse
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import math
import pandas as pd
from torchvision import transforms

# Analyze command-line parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Training loop for select_width_size.")
    
    # 参数设置
    parser.add_argument('--data_path', type=str, required=True, help="Path: Training data storage location")
    parser.add_argument('--csv_path', type=str, required=True, help="Path: CSV file for saving accuracy and loss values")
    parser.add_argument('--weights_path_template', type=str, required=True, help="Path: Template for saving model weights")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of classes")
    parser.add_argument('--batch_size', type=int, default=128, help="The number of samples per batch")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training rounds")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--device', default='cuda:0', help="Device number (e.g. 0 or CPU)")
    
    args = parser.parse_args()
    return args

# Model and Training Code
def main():
    # Get command-line parameters
    args = parse_args()

    # Determine equipment
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    

    tb_writer = SummaryWriter()

    # Read data
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # Data transformation
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # Create training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Create validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # Load data
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

     
    # Parameters for CAF model
    caf_params = {
        'image_size': 64,  # 1/2 real image input size
        'patch_size': 8,
        'num_classes': 6,  # Default value, will be modified dynamically
        'dim': 512,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 1024,
        'dim_head': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'mode': 'CAF'
    }

    # change caf_params 的 num_classes
    caf_params['num_classes'] = args.num_classes

    # initial GARN_CAF model
    model_GARN_CAF = GARN_CAF(
        garn_in_channels=24,
        garn_num_blocks=4,
        garn_n_feats=32,
        garn_n_factors=4,
        caf_params=caf_params
    ).to(device)

    # Set optimizer
    optimizer = optim.SGD(model_GARN_CAF.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # Learning rate scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.05) + 0.05
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Ensure the directory exists
    weights_dir = args.weights_path_template
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    #Column Name
    df = pd.DataFrame(columns=['time','epoch','train loss','train acc','valid loss','valid acc','New Start']) 
    df.to_csv(args.csv_path,mode='a',index=False)

    # Training cycle
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model_GARN_CAF, optimizer, train_loader, device, epoch)
        scheduler.step()

        val_loss, val_acc = evaluate(model_GARN_CAF, val_loader, device, epoch)

        # Record the results of training and validation
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        
        # Save Model
        # Save the model with the correct file name
        weights_path = os.path.join(weights_dir, "model-{}.pth".format(epoch))
        torch.save(model_GARN_CAF.state_dict(), weights_path)

        # Store to CSV
        time = str("%s" % datetime.now())
        data = pd.DataFrame([[time, epoch+1, train_loss, train_acc, val_loss, val_acc]])
        data.to_csv(args.csv_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()



      
        




