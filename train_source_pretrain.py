import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
import os
from time import time
from tqdm import tqdm
import Data.dataset as dataset
from Data.dataset import minmax_scaler
from Utils.fourier import *
from Utils.utils import GradientReversal,loop_iterable
from Utils.transform import intensity_transform
import fft_model
import GPUtil
import torch.cuda
import random
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
random.seed(random_seed)
np.random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="4")
    # gpu id number

    parser.add_argument('--model_name', type=str, default='source_pretrain')
    # Model save name

    parser.add_argument('--batch_size', type=int, default=4)
    # batch size
    parser.add_argument('--init_lr', type=float, default=1e-4)
    # learning rate
    parser.add_argument('--epochs', type=int, default=50)
    # number of epochs
    parser.add_argument('--task', type=str, default='adcn')
    # task : adcn, admci, cnmci
    parser.add_argument('--task_num', type=str, default='task')
    # task : adcn / task2 : admci / task3 : cnmci
    parser.add_argument('--source', type=str, default='aibl')
    # source domain

    args, _ = parser.parse_known_args()
    return args

args = parse_args()

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print(torch.cuda.is_available())

data_path_int = "/DataRead/ysshin/{}/{}_data_{}_int.npy".format(args.task_num, args.source, args.task)
data_path = "/DataRead/ysshin/{}/{}_data_{}.npy".format(args.task_num, args.source, args.task)
label_path ="/DataRead/ysshin/{}/{}_label_{}.npy".format(args.task_num, args.source, args.task)

dataset = dataset.INTDataset(data_path=data_path, data_path_int=data_path_int, label_path=label_path, transform=None)


shuffled_indices = np.random.permutation(len(dataset))
train_idx = shuffled_indices[:int(0.8*len(dataset))]
val_idx = shuffled_indices[int(0.8*len(dataset)):]

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True,
                                          sampler=SubsetRandomSampler(train_idx))
    
val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx))

print('Data loading done...\n')
net = fft_model.Net(dropout=0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

feature_extractor = net.feature_extractor
clf = net.classifier

intensity_classifier = nn.Sequential(
    GradientReversal(lambda_= 0.1),
    nn.Linear(128*2*3*2, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)


optimizer = torch.optim.Adam(list(intensity_classifier.parameters()) + list(net.parameters()), lr=args.init_lr)
loss_function = nn.CrossEntropyLoss()

def do_epoch(model, dataloader, criterion, optim = None):
    
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x_ori, y_true, xi in tqdm(dataloader, leave=False):
        xi = apr(x_ori, xi, ratio=1.0)
        
        x_ori = minmax_scaler(x_ori)
        xi = minmax_scaler(xi)

        x = torch.cat([x_ori, xi])

        x, y_true = x.to(device), y_true.to(device)

        intensity_y = torch.cat([torch.ones(x_ori.shape[0]), torch.zeros(xi.shape[0])])
        intensity_y = intensity_y.to(device)

        features = feature_extractor(x).view(x.shape[0], -1)
        intensity_preds = intensity_classifier(features).squeeze()
        label_pred = clf(features[:x_ori.shape[0]])
        intensity_loss = F.binary_cross_entropy_with_logits(intensity_preds, intensity_y)
        label_loss = F.cross_entropy(label_pred, y_true)

        _, preds = torch.max(label_pred.data, 1)
        total_size += y_true.size(0)

        loss = label_loss + intensity_loss

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_acc += (preds == y_true).sum().item()

    mean_loss = total_loss / len(dataloader)
    mean_acc = total_acc / total_size

    return mean_loss, mean_acc

best_loss = 100000
best_acc = 0

for epoch in range(0, args.epochs):
    net.train()
    train_loss, train_acc = do_epoch(net, train_loader, loss_function, optim=optimizer)

    net.eval()
    with torch.no_grad():
        val_loss, val_acc = do_epoch(net, val_loader, loss_function, optim=None)

    tqdm.write(f'Epoch {epoch:03d}: train_loss = {train_loss:.4f} , train_acc = {train_acc:.4f}')
    tqdm.write(f'val_loss = {val_loss: .4f} , val_acc = {val_acc:.4f}')

    if val_acc > best_acc:
        tqdm.write(f'Saving model... Selection: val_acc')
        best_acc = val_acc
        torch.save(net.state_dict(), "models/pretrain/{}/{}_acc.pt".format(args.task, args.model_name))

print("Train Finished!!")
