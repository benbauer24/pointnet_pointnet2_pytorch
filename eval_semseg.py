import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from torch.autograd import Variable
from data_utils.S3DISDataLoader import S3DISDataLoader, recognize_all_data,class2label
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test_semseg, save_checkpoint
from model.pointnet2 import PointNet2SemSeg
from model.pointnet import PointNetSeg, feature_transform_reguliarzer

seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

checkpoint_file = "./experiment/pointnet2SemSeg-2022-05-26_05-29/checkpoints/pointnet2_004_0.8348.pth"

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=12, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_file, help='checkpoint')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of model')
    return parser.parse_args()

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(
        str(experiment_dir) + '/%sSemSeg-' % args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/eval_%s_semseg.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        '---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = recognize_all_data(test_area=5)
    test_dataset = S3DISDataLoader(test_data, test_label)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize,
                                                 shuffle=True, num_workers=int(args.workers))

    '''MODEL LOADING'''
    num_classes = 13
    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointNet2SemSeg(num_classes) if args.model_name == 'pointnet2' else PointNetSeg(num_classes,
                                                                                            feature_transform=True,
                                                                                            semseg=True)
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        logger.info('Load CheckPoint')
        checkpoint = torch.load(args.checkpoint)
        #start_epoch = checkpoint['epoch']
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(torch.load(checkpoint))
    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)
        start_epoch = 0

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    model = model.eval()
    iou_tabel = np.zeros((len(catdict), 3))
    metrics = defaultdict(lambda: list())
    hist_acc = []
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        points, target = data
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        with torch.no_grad():
            if pointnet2:
                pred = model(points[:, :3, :], points[:, 3:, :])
            else:
                pred, _ = model(points)
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batchsize * num_point))
    iou_tabel[:, 2] = iou_tabel[:, 0] / iou_tabel[:, 1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    print(metrics['accuracy'])
    iou_tabel = pd.DataFrame(iou_tabel, columns=['iou', 'count', 'mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict))]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    print(cat_iou)
    logger.info(cat_iou)

    mean_iou = np.mean(cat_iou)
    print('Total accuracy: %f  meanIOU: %f' % (test_metrics['accuracy'], mean_iou))
    logger.info('Test accuracy: %f  meanIOU: %f' % (test_metrics['accuracy'], mean_iou))

    logger.info('End of evaluation...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
