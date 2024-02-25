import os
import numpy as np
import random
from tqdm import tqdm

# 设置随机种子
random.seed(42)


def save_slices_from_npz(source_folder, output_folder):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有以.npz结尾的文件
    npz_files = [f for f in os.listdir(source_folder) if f.endswith('.npz')]
    # 计算文件数量的一半
    half_count = len(npz_files) // 2
    # 随机选择一半的文件
    selected_files = random.sample(npz_files, half_count)

    for file in tqdm(selected_files, desc="Processing files"):
        file_path = os.path.join(source_folder, file)
        # 加载npz文件
        data = np.load(file_path)['data']
        # 检查数据维度 (应该是[C,Z,H,W])
        if len(data.shape) == 4:
            # 遍历Z轴上的每个切片
            for z in range(data.shape[1]):
                # 提取切片
                slice_data = data[:, z, :, :]
                # 格式化切片编号
                slice_num = f'slice{z:03d}'
                # 创建新的文件名
                new_file_name = f"{file.replace('.npz', '') + slice_num}.npz"
                # 保存切片
                np.savez(os.path.join(output_folder, new_file_name), data=slice_data)

            # 删除原始npz文件
            os.remove(file_path)
        else:
            print(f"File {file} has an unexpected shape and was skipped.")


if __name__ == "__main__":
    datasets = {
        'amos': {'source': r"J:\datasets\testing\Task202_AMOSTest\nnUNetData_plans_v2.1_stage0", 'output': './val_slices_amos'},
        'flare': {'source': r"J:\datasets\testing\Task204_FLARE22Test\nnUNetData_plans_v2.1_stage0", 'output': './val_slices_flare'},
        'word': {'source': r"J:\datasets\testing\Task203_WORDTest\nnUNetData_plans_v2.1_stage0", 'output': './val_slices_word'},
        # Add more datasets if needed
    }

    for dataset_name, paths in datasets.items():
        print(f"Processing {dataset_name}...")
        save_slices_from_npz(paths['source'], paths['output'])
