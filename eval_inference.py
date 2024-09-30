import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import dataset
from model import Net
from sklearn.metrics import confusion_matrix, roc_auc_score
#from transform import source_transform, adapt_transform
from dataset import minmax_scaler
from tqdm import tqdm
import GPUtil
import random
import fft_model
import argparse

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
random.seed(random_seed)
np.random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="0")
    # gpu id number

    parser.add_argument('--model_name', type=str, default='source_disentangle')
    # model save name

    parser.add_argument('--batch_size', type=int, default=1)
    # batch size
    parser.add_argument('--test_target', type=str, default='adni3')
    # target name
    parser.add_argument('--task', type=str, default='adcn')
    # task name
    parser.add_argument('--task_num', type=str, default='task')
    # step
    parser.add_argument('--step', type=str, default='adapt')

    args, _ = parser.parse_known_args()
    return args

args = parse_args()

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

target_path = "/DataRead/ysshin/{}/data_8_2/{}_data_{}_82_test.npy".format(args.task_num, args.test_target, args.task)
trg_label_path = "/DataRead/ysshin/{}/data_8_2/{}_label_{}_82_test.npy".format(args.task_num, args.test_target, args.task)

dataset = dataset.MyDataset(data_path = target_path, label_path = trg_label_path, transform=None)
dataloader_target = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

MODEL_FILE = 'models/{}/{}_acc.pt'.format(args.task, args.model_name)

net = fft_model.Net(dropout=0.5)
net.to(device)
net.load_state_dict(torch.load(MODEL_FILE))
net.eval()
print('Test a model on the target domain...')
correct = 0
total = 0

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in tqdm(dataloader_target):
        images = minmax_scaler(images)
        images = images.to(device)
        labels = labels.to(device)

        #outputs, _ = net(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        y_pred.append(predicted.cpu().detach().numpy())
        y_true.append(labels.cpu().detach().numpy())

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = (tp+tn) / (tp+tn+fp+fn)
sensitivity = tp / (tp+fn)
specificity = tn / (tn+fp)

print('###########################################\n')
print('Test : {}'.format(MODEL_FILE))
print('Accuracy(ACC) on the target domain : ', accuracy)
print('Sensitivity(SEN) on the target domain : ', sensitivity)
print('Specificity(SPE) on the target domain : ', specificity)
print('AUC curve(AUC) on the target domain : ', roc_auc_score(y_true, y_pred))
print('\n###########################################')