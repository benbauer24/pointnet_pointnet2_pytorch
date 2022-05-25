import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=24, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--model_name', default='pointnet2', help='model name')
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # datapath = './data/ModelNet/'
    datapath = "/media/bedad/DATA/ETS/Recherche/programmation/data/pointnet_pytorch/ModelNet40/"

    ROTATION = None

    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./eval_experiment/checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = Path('./eval_experiment/logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("PointNet2")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./eval_experiment/logs/eval_%s_' % args.model_name + str(
        datetime.datetime.now().strftime('%Y-%m-%d %H-%M')) + '.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        '---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    logger.info("The number of test data is: %d", test_data.shape[0])
    testDataset = ModelNetDataLoader(test_data, test_label, rotation=ROTATION)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''MODEL LOADING'''
    num_class = 40
    classifier = PointNetCls(num_class,
                             args.feature_transform).cuda() if args.model_name == 'pointnet' else PointNet2ClsMsg().cuda()
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        logger.info('Load CheckPoint')
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)
        start_epoch = 0

    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    classifier = classifier.eval()
    mean_correct = []
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        pointcloud, target = data
        target = target[:, 0]

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()

        mean_correct.append(correct.item()/float(points.size()[0]))

    accuracy = np.mean(mean_correct)
    print('Total Accuracy: %f'%accuracy)

    logger.info('Total Accuracy: %f'%accuracy)
    logger.info('End of evaluation...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
