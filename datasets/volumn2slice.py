import os
import numpy as np
from tqdm import tqdm


def process_and_save_slices(input_folder_path, output_folder_path):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 遍历指定文件夹中的所有文件
    for file in tqdm(os.listdir(input_folder_path)):
        # 检查文件扩展名是否为.npz
        if file.endswith('.npz'):
            file_path = os.path.join(input_folder_path, file)
            # 加载npz文件
            data = np.load(file_path)['data']

            # 遍历Z轴上的每个切片
            for z in range(data.shape[1]):
                # 提取切片
                slice_data = data[:, z, :, :]
                # 创建新文件名
                new_file_name = '{}slice{:03d}.npz'.format(file[:-4], z)
                new_file_path = os.path.join(output_folder_path, new_file_name)

                # 检查new_file_path的父目录是否存在，如果不存在则创建
                new_file_dir = os.path.dirname(new_file_path)
                if not os.path.exists(new_file_dir):
                    os.makedirs(new_file_dir)

                # 保存切片到新文件
                np.savez(new_file_path, data=slice_data)


if __name__ == '__main__':
    # 创建一个字典，键为输入路径，值为输出路径
    datasets_dict = {
        # r"J:\datasets\medical\dataset\nnUNet_preprocessed\Task201_FLARE22labeled\nnUNetData_plans_v2.1_stage0":
        #     'data/output/flare22',
        # r"J:\datasets\medical\dataset\nnUNet_preprocessed\Task203_WORD\nnUNetData_plans_v2.1_stage0":
        #     'data/output/word',
        # r"J:\datasets\medical\dataset\nnUNet_preprocessed\Task218_AMOS2022_postChallenge_task1\nnUNetData_plans_v2.1_stage0":
        #     'data/output/amos',
        r"J:\datasets\medical\dataset\nnUNet_preprocessed\Task205_ALTAS\nnUNetData_plans_v2.1_stage0":
            'data/output/altas',
        r"J:\datasets\medical\dataset\nnUNet_preprocessed\Task219_AMOS2022_postChallenge_MR\nnUNetData_plans_v2.1_stage0":
            'data/output/amos_mr',
        'data/val/val_slices_altas_3d': 'data/val/val_slices_altas',
        'data/val/val_slices_amos_mr_3d': 'data/val/val_slices_amos_mr',
        # ...可以继续添加更多的路径对...
    }

    # 遍历字典，处理每个数据集
    for input_path, output_path in datasets_dict.items():
        process_and_save_slices(input_path, output_path)
