import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import torchvision.models as models
import matplotlib.pyplot as plt
from patch_attention import att_lstm
from torchsummary import summary
import time
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, classification_report
from tqdm import tqdm as tqdm
from network import resnet
from collections import OrderedDict
from network import sa_resnet
from patch_transformer import att_lstm
from transformer_wzw.transformer import visual_prompt

parser = argparse.ArgumentParser(description='lstm aggregator classifier test script')
parser.add_argument('--lib', type=str, default='', help='path to MIL library binary')
parser.add_argument('--output', type=str, default='', help='name of output file')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 512)')
parser.add_argument('--lstm', default='ATTLSTM_checkpoint_best.pth', type=str, help='path to RNN checkpoint')
parser.add_argument('--k', default=10, type=int, help='top k tiles assumed to be the same class as slide (1: standard MIL)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def Parallel2Single(origin_state):
    converted = OrderedDict()

    for k, v in origin_state.items():
        name = k[7:]
        converted[name] = v

    return converted


def main():
    global args
    args = parser.parse_args()
    # cnn
    model = sa_resnet.sa_resnet50(False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model_dict = torch.load('SaCNN_checkpoint_best.pth')

    model.cuda()
    model.load_state_dict(Parallel2Single(model_dict['state_dict']))
    get_feature_model = nn.Sequential(*list(model.children())[:-1]).cuda()
    get_feature_model.eval()

    lstm_model = att_lstm(2048, 256, 2, True, 2).cuda()            # 156
    lstm_dict = torch.load(args.lstm)
    lstm_model.load_state_dict(lstm_dict['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    test_metric_probs_inf_save = torch.load('output_Sa/test_metric_probs_inf2.db')

    test_dset = MILdataset(args.lib, args.k, trans,
                            test_metric_probs_inf_save['test_dset_grid'],
                            test_metric_probs_inf_save['test_dset_slideIDX'])
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=False)
    t_probs = test_metric_probs_inf_save['test_probs']
    t_topk = group_argtopk(np.array(test_dset.slideIDX), t_probs, args.k)                   # 1
    test_dset.setmode(3)
    test_dset.settopk(t_topk, get_feature_model)
    time_mark = time.strftime('%Y_%m_%d_', time.localtime(time.time()))

    probs = inference(test_loader, lstm_model, args.batch_size)
    pred = []

    fconv = open(os.path.join(args.output, time_mark + 'attlstm2_predictions.csv'), 'w')
    fconv.write(' file,target,prediction,probability\n')
    for name, target, prob in zip(test_dset.slidenames, test_dset.targets, probs):
        fconv.write('{},{},{},{}\n'.format(name, target, int(prob >= 0.5), prob))
        pred.append(int(prob >= 0.5))
    fconv.close()

    metrics_meters = calc_accuracy(pred, test_dset.targets)
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
    s = ', '.join(str_logs)
    print('Test  metrics: ' + s)
    plot_ROC(test_dset.targets, pred)


def calc_accuracy(pred, real):
    if str(type(pred)) != "<class 'numpy.ndarray'>":
        pred = np.array(pred)
    if str(type(real)) != "<class 'numpy.ndarray'>":
        real = np.array(real)
    neq = np.not_equal(pred, real)
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    balanced_acc = balanced_accuracy_score(real, pred)
    recall = recall_score(real, pred, average='weighted')
    metrics_meters = {'acc': round(balanced_acc, 4), 'recall': round(recall, 4), 'fnr': round(fnr, 4)}
    label_list = ["0", "1"]
    cr = classification_report(real, pred, target_names=label_list, digits=3)
    print(cr, "\n")

    cm = confusion_matrix(real, pred)
    plt.title('Confusion matrix', size=15)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, label_list, size=12)
    plt.yticks(tick_marks, label_list, size=12)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    thresh = cm.max() / 2
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.text(y, x, cm[x][y], horizontalalignment='center', fontsize=20 , verticalalignment='center',
                     color="white" if cm[x][y] > thresh else "black")
    plt.show()
    return metrics_meters


def plot_ROC(y, prob):
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', size=19)
    plt.ylabel('True Positive Rate', size=19)
    plt.title('ROC curve', size=22)
    plt.legend(loc="lower right", fontsize='xx-large')
    plt.show()


def inference(loader, model, batch_size):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))

    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            print('Testing - Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+batch_size] = output.detach()[:, 1].clone()

    return probs.cpu().numpy()


def group_argtopk(groups, data, k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', k=0, transform=None, load_grid=None, load_IDX=None):
        lib = torch.load(libraryfile)
        slides = []
        patch_size = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
            patch_size.append(int(lib['patch_size'][i]))
        print('')
        if load_IDX is None:
            grid = []
            slideIDX = []
            for i, g in enumerate(lib['grid']):
                if len(g) < k:
                    g = g + [(g[x]) for x in np.random.choice(range(len(g)), k-len(g))]               
                grid.extend(g)    
                slideIDX.extend([i]*len(g))
        else:
            grid = load_grid
            slideIDX = load_IDX
        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.patch_size = patch_size
        self.level = lib['level']

    def setmode(self, mode):
        self.mode = mode
        
    def settopk(self, top_k=None, feature_extract_model=None):
        self.top_k = top_k
        self.feature_extract_model = feature_extract_model

    def maketraindata(self, idxs, repeat=0):
        if abs(repeat) == 0:
            self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0) for x in idxs]
        else:
            repeat = abs(repeat) if repeat % 2 == 1 else abs(repeat) + 1
            self.t_data = [(self.sliall_traindeIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0) for x in idxs]
            for y in range(-100,int(100 + repeat/2),int(100*2/repeat)):
                self.t_data = self.t_data + [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]], y/1000) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.patch_size[slideIDX],\
                                                    self.patch_size[slideIDX])).convert('RGB')
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target,h_value = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.patch_size[slideIDX],\
                                                    self.patch_size[slideIDX])).convert('RGB')
            if h_value > 0:
                hue_factor = random.uniform(h_value,0.1)
            elif h_value == 0:
                hue_factor = random.uniform(0,0)
            elif h_value < 0:
                hue_factor = random.uniform(-0.1,h_value)
            img = functional.adjust_hue(img,hue_factor)
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        elif self.mode == 3 and self.top_k is not None and self.feature_extract_model is not None:
            k_value = int(len(self.top_k)/len(self.targets))
            for j in range(k_value):
                coord = self.grid[self.top_k[index * k_value + j]]
                img = self.slides[index].read_region(coord, self.level, (self.patch_size[index], self.patch_size[index])).convert('RGB')
                if img.size != (224, 224):
                    img = img.resize((224, 224), Image.BILINEAR)
                img = self.transform(img).unsqueeze(0)
                if j == 0:
                    feature = self.feature_extract_model(img.cuda())
                else:
                    feature = torch.cat((feature, self.feature_extract_model(img.cuda())), 0)
            return feature.view(-1, feature.shape[1]), self.targets[index]
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        elif self.mode == 3 and self.top_k is not None and self.feature_extract_model is not None:
            return len(self.targets)

if __name__ == '__main__':
    main()

