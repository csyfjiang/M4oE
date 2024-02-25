import os
import csv

import numpy as np
import tqdm as tqdm

import os
import csv
import numpy as np

def npz_csv(system='Linux',csv_task_name='datasets_v217'):
    datasets_config = {}
    if system == 'Windows':
        dataset_config_windows = {
            # 'Flare22': {
            #     # 'data_dir': 'datasets/slice/flare22',
            #     'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\flare22',
            #     'img_idx': 0,
            #     'label_idx': 1,
            #     'dataset_id': 0,
            #     'num_classes': 14,
            #     'predict_head': 0
            # },
            # 'AMOS': {
            #     # 'data_dir': 'datasets/data/slice/amos',
            #     'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\amos',
            #     'img_idx': 0,
            #     'label_idx': 1,
            #     'dataset_id': 1,
            #     'num_classes': 16,
            #     'predict_head': 1
            # },
            # # 'WORD': {
            # #     # 'data_dir': 'datasets/data/slice/word',
            # #     'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\word',
            # #     'img_idx': 0,
            # #     'label_idx': 1,
            # #     'dataset_id': 2,
            # #     'num_classes': 17,
            # #     'predict_head': 2
            # # },
            # 'ALTAS': {
            #     'data_dir': 'datasets/data/slice/altas',
            #     # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\word',
            #     'img_idx': 0,
            #     'label_idx': 1,
            #     'dataset_id': 2,
            #     'num_classes': 3,
            #     'predict_head': 2
            # },
            # 'AMOS_MR': {
            #     'data_dir': 'datasets/data/slice/amos_mr',
            #     # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\amos',
            #     'img_idx': 0,
            #     'label_idx': 1,
            #     'dataset_id': 3,
            #     'num_classes': 16,
            #     'predict_head': 3
            # },
        }
        datasets_config = dataset_config_windows
    elif system == 'Linux':
        # dataset_config_linux = {
        #     'Flare22': {
        #         'data_dir': 'datasets/slice/flare22',
        #         # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\flare22',
        #         'img_idx': 0,
        #         'label_idx': 1,
        #         'dataset_id': 0,
        #         'num_classes': 14,
        #         'predict_head': 0
        #     },
        #     'AMOS': {
        #         'data_dir': 'datasets/slice/amos',
        #         # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\amos',
        #         'img_idx': 0,
        #         'label_idx': 1,
        #         'dataset_id': 1,
        #         'num_classes': 16,
        #         'predict_head': 0
        #     },
        #     'WORD': {
        #         'data_dir': 'datasets/slice/word',
        #         # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\word',
        #         'img_idx': 0,
        #         'label_idx': 1,
        #         'dataset_id': 2,
        #         'num_classes': 17,
        #         'predict_head': 0
        #     },
        # }
        dataset_config_linux = {
            'Flare22': {
                'data_dir': 'datasets/slice/flare22',
                # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\flare22',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 0,
                'num_classes': 14,
                'predict_head': 0
            },
            # 'AMOS': {
            #     'data_dir': 'datasets/slice/amos',
            #     # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\amos',
            #     'img_idx': 0,
            #     'label_idx': 1,
            #     'dataset_id': 1,
            #     'num_classes': 16,
            #     'predict_head': 1
            # },
            #  'WORD': {
            #     # 'data_dir': 'datasets/data/slice/word',
            #     'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\word',
            #     'img_idx': 0,
            #     'label_idx': 1,
            #     'dataset_id': 2,
            #     'num_classes': 17,
            #     'predict_head': 2
            # },
            'ALTAS': {
                'data_dir': 'datasets/slice/altas',
                # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\word',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 1,
                'num_classes': 3,
                'predict_head': 1
            },
            'AMOS_MR': {
                'data_dir': 'datasets/slice/amos_mr',
                # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\slice\amos',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 2,
                'num_classes': 16,
                'predict_head': 2
            },
        }
        datasets_config = dataset_config_linux
    # csv_file_path = './lists/datasets_v10.csv'
    csv_file_path = f'./lists/{csv_task_name}.csv'
    # 确保CSV文件的目录存在
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # 打开CSV文件进行写入
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # 写入CSV文件头
        writer.writerow(["data_dir", "img_idx", "label_idx", "dataset_id", "predict_head", "n_classes"])

        # 遍历配置中的每个数据集
        for dataset_name, config in datasets_config.items():
            # 列出data_dir中所有的.npz文件
            data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.npz')]

            # 为找到的每个.npz文件写入一行
            for npz_file in data_files:
                npz_file_path = os.path.join(config['data_dir'], npz_file)

                writer.writerow([
                    npz_file_path,  # .npz文件的完整路径
                    config['img_idx'],  # 图像索引
                    config['label_idx'],  # 标签索引
                    config['dataset_id'],  # 数据集ID
                    config['predict_head'],  # 预测头
                    config['num_classes'],  # 类别数量
                ])

def nifiti_csv():
    dataset_config = {
        'Flare22': {
            # 'img_path': 'imgdata/flare22/images',
            # 'label_path': 'imgdata/flare22/labels',
            'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\FLARE22\Train\Training-20230908T070644Z-002\Training\FLARE22_LabeledCase50\images',
            'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\FLARE22\Train\Training-20230908T070644Z-002\Training\FLARE22_LabeledCase50\labels',
            'dataset_id': 0,
            'num_classes': 14,
            'predict_head': 0
        },
        'AMOS': {
            # 'img_path': 'imgdata/amos/images',
            # 'label_path': 'imgdata/amos/labels',
            'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\amos22\amos22\imagesTr',
            'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\amos22\amos22\labelsTr',
            'dataset_id': 1,
            'num_classes': 16,
            'predict_head': 1
        },
        'WORD': {
            # 'img_path': 'imgdata/word/images',
            # 'label_path': 'imgdata/word/labels',
            'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\WORD-V0.1.0\WORD-V0.1.0\imagesTr',
            'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\WORD-V0.1.0\WORD-V0.1.0\labelsTr',
            'dataset_id': 2,
            'num_classes': 17,
            'predict_head': 2
        }
    }

    with open('./lists/datasets_v2.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["image", "label", "dataset_id", "predict_head", "n_classes"])  # Add n_classes column

        for dataset, config in dataset_config.items():
            img_path = config['image_dir']
            label_path = config['label_dir']
            dataset_id = config['dataset_id']
            num_classes = config['num_classes']  # Get num_classes from the config
            predict_head = config['predict_head']  # Get predict_head from the config

            # Get image and label filenames that end with .nii.gz
            img_filenames = sorted([f for f in os.listdir(img_path) if f.endswith('.nii.gz')])
            label_filenames = sorted([f for f in os.listdir(label_path) if f.endswith('.nii.gz')])

            # Check if the number of images and labels match
            if len(img_filenames) != len(label_filenames):
                print(f"Warning: The number of images and labels in {dataset} do not match.")
                continue

            # Write each image and corresponding label to the CSV
            for img_filename, label_filename in zip(img_filenames, label_filenames):
                img_file_path = os.path.join(img_path, img_filename)
                label_file_path = os.path.join(label_path, label_filename)
                writer.writerow(
                    [img_file_path, label_file_path, dataset_id, predict_head, num_classes])  # Include num_classes

def npz_csv_testing():
    dataset_config_test = {
        # 'Flare22': {
        #     'data_dir': "datasets/val/val_slices_flare",
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 0,
        #     'num_classes': 14,
        #     'predict_head': 0
        # },
        # 'AMOS': {
        #     'data_dir': "datasets/val/val_slices_amos",
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 1,
        #     'num_classes': 16,
        #     'predict_head': 1
        # },
        # 'WORD': {
        #     'data_dir':"datasets/val/val_slices_word",
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 2,
        #     'num_classes': 17,
        #     'predict_head': 2
        # }
        # 'ALTAS': {
        #     # 'data_dir': 'datasets/val/val_slices_altas',
        #     'data_dir': r"J:\datasets\testing\Task205_ALTAS",
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 1,
        #     'num_classes': 3,
        #     'predict_head': 1
        # },
        'AMOS_MR': {
            # 'data_dir': 'datasets/val/val_slices_amos_mr',
            'data_dir': r"J:\datasets\testing\Task206_AMOS_mr",
            'img_idx': 0,
            'label_idx': 1,
            'dataset_id': 2,
            'num_classes': 16,
            'predict_head': 2
        },
    }# autodl-tmp/project/
    dataset_config_val = {
        # 'Flare22': {
        #     # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\val\val_slices_flare',
        #     'data_dir': 'datasets/remap/flare_val',
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 0,
        #     'num_classes': 22,
        #     'predict_head': 0
        # },
        # 'AMOS': {
        #     # 'data_dir': r'D:\Research\Swin-Unet\datasets\data\val\val_slices_amos',
        #     'data_dir': 'datasets/remap/amos_val',
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 0,
        #     'num_classes': 22,
        #     'predict_head':0
        # },
        # 'WORD': {
        #     # 'data_dir':r'D:\Research\Swin-Unet\datasets\data\val\
        #     'data_dir': 'datasets/remap/word_val',
        #     'img_idx': 0,
        #     'label_idx': 1,
        #     'dataset_id': 0,
        #     'num_classes': 22,
        #     'predict_head': 0
        # }
    }
    csv_file_path = './lists/datasets_multi_test_amos_mr_moe.csv'

    # Ensure the directory for the CSV file exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # Write the header row with an additional 'num_slices' column
        writer.writerow(["data_dir", "img_idx", "label_idx", "dataset_id", "predict_head", "n_classes"])

        # Iterate over each dataset in the configuration
        for dataset_name, config in dataset_config_test.items():
            # List all .npz files in the data_dir
            data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.npz')]

            # Write a row for each .npz file found
            for npz_file in data_files:
                npz_file_path = os.path.join(config['data_dir'], npz_file)
                # Load the .npz file to find out the number of slices
                npz_data = np.load(npz_file_path)

                writer.writerow([
                    npz_file_path,  # Full path to the .npz file
                    config['img_idx'],  # Image index
                    config['label_idx'],  # Label index
                    config['dataset_id'],  # Dataset ID
                    config['predict_head'],  # Prediction head
                    config['num_classes'],  # Number of classes
                ])
if __name__ == "__main__":
    # npz_csv(system='Linux',csv_task_name='datasets_v218')
    npz_csv_testing()
    # nifiti_csv()