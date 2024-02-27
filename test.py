import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
### modify
parser = argparse.ArgumentParser()
# parser.add_argument('--volume_path', type=str,
#                     default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='amos', help='experiment_name')
# parser.add_argument('--num_classes', type=int,
#                     default=9, help='output channel of network')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--data_csv', type=str,
                    default='./lists/datasets_test.csv',
                    help='path to the dataset csv file'
                    )

parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(csv_file=args.data_csv, modes='test', transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        dataset_id, predict_head, n_classes = sampled_batch["dataset_id"], sampled_batch["predict_head"], sampled_batch["n_classes"]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume(image, label, model,dataset_id=dataset_id,predict_head=predict_head, classes=n_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        # print(metric_i)
        # print('test script,',type(metric_i))
        metric_list += np.array(metric_i, dtype=object)
        # print('metric_list',metric_list)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_iou %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(db_test)
    for i in range(1, n_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_iou %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_iou : %f' % (performance, mean_hd95, mean_iou))
    return "Testing Finished!"

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'flare22': {
            'Dataset': Synapse_dataset,
            'root_path': './',
            'data_csv': './lists/datasets_flaretest.csv',
            'dataset_id': 0,
            'num_classes': 14,
            "predict_head": 0,
            'z_spacing': 1
        },
        'amos': {
            'Dataset': Synapse_dataset,
            'root_path': './',
            'data_csv': './lists/datasets_amostest.csv',
            'dataset_id': 1,
            'num_classes': 16,
            "predict_head": 0,
            'z_spacing': 1
        },
        'word': {
            'Dataset': Synapse_dataset,
            'root_path': './',
            'data_csv': './lists/datasets_wordtest.csv',
            'dataset_id': 2,
            'num_classes': 17,
            "predict_head": 0,
            'z_spacing': 1
        },
        # 'remap': {
        #     'Dataset': Synapse_dataset,
        #     'z_spacing': 1,
        #     'num_classes': 22,
        # },

    }


    num_classes_per_dataset = [14, 3, 16]
    # num_classes_per_dataset = [dataset_config[name]['num_classes'] for name in dataset_config]
    dataset_name = args.dataset
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    # args.data_csv = dataset_config[dataset_name]['data_csv']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=num_classes_per_dataset).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_multimodal'
    os.makedirs(log_folder, exist_ok=True)

    snapshot_dir = os.path.join(log_folder, args.output_dir)
    os.makedirs(snapshot_dir, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


