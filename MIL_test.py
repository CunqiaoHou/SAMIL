import sys
import os
import numpy as np
import argparse
import random
import time
import openslide
import PIL.Image as Image
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
# import torchvision.models as models
from network import resnet, mpncovresnet
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import balanced_accuracy_score,recall_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, classification_report
from network import sa_resnet


parser = argparse.ArgumentParser(description='')
parser.add_argument('--lib', type=str, default='', help='path to data file')
parser.add_argument('--output', type=str, default='', help='name of output directory')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=64, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 4)')


def main():
    global args
    args = parser.parse_args()

    # load model
    model = sa_resnet.sa_resnet50(False)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.cuda()
    model = nn.DataParallel(model)
    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    summary(model, (3, 224, 224))
    cudnn.benchmark = True

    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    # load data
    dset = MILdataset(args.lib, trans)
    loader = torch.utils.data.DataLoader(dset,batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers, pin_memory=False)
    test_metric_probs_inf_save2 = {'test_dset_slideIDX': dset.slideIDX, 'test_dset_grid': dset.grid}
    time_mark = time.strftime('%Y_%m_%d_', time.localtime(time.time()))
    start_time = time.time()
    dset.setmode(1)
    probs = inference(loader, model, args.batch_size, 'test')
    maxs = group_max(np.array(dset.slideIDX), probs, len(dset.targets))
    pred = [1 if x >= 0.5 else 0 for x in maxs]
    metrics_meters = calc_accuracy(pred, dset.targets)
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
    s = ', '.join(str_logs)
    print('Test  metrics: ' + s)
    # 绘制ROC曲线
    plot_ROC(dset.targets, pred)
    # 备份所有预测结果
    fp = open(os.path.join(args.output, time_mark + 'Test_SaCNN_predictions_2.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, maxs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob >= 0.5), prob))
    fp.close()

    test_metric_probs_inf_save2['test_probs'] = probs.copy()
    torch.save(test_metric_probs_inf_save2, 'output_Sa/external/ex_test_metric_probs_inf224.db')
    print('test has been finished, needed {} sec.'.format((time.time() - start_time)))


def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))

    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i + 1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i * args.batch_size:i * args.batch_size + input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def calc_accuracy(pred, real):
    if str(type(pred)) != "<class 'numpy.ndarray'>":
        pred = np.array(pred)
    if str(type(real)) != "<class 'numpy.ndarray'>":
        real = np.array(real)
    neq = np.not_equal(pred, real)
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    balanced_acc = balanced_accuracy_score(real, pred)
    recall = recall_score(real, pred, average='weighted')
    metrics_meters = {'acc': round(balanced_acc, 3), 'recall': round(recall, 3), 'fnr': round(fnr, 3)}
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
            plt.text(y, x, cm[x][y], horizontalalignment='center', verticalalignment='center',
                     color="white" if cm[x][y] > thresh else "black")
    plt.show()
    return metrics_meters


def plot_ROC(y, prob):          # 绘制ROC和AUC，来判断模型的好坏
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', size=19)
    plt.ylabel('True Positive Rate', size=19)
    plt.title('ROC curve', size=22)
    plt.legend(loc="lower right", fontsize='xx-large')
    plt.show()


def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return list(out)


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        patch_size = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
            patch_size.append(int(lib['patch_size'][i]))
        print('')
        # Flatten grid
        grid = []
        slideIDX = []
        for i, g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.size = patch_size
        self.level = lib['level']

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size[slideIDX], self.size[slideIDX])).convert('RGB')
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size[slideIDX], self.size[slideIDX])).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)


if __name__ == '__main__':
    main()
