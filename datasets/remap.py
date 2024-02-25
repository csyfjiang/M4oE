import os
import numpy as np
from tqdm import tqdm

# 标签映射字典
label_mappings = {
    'flare': {
        1: 1, 13: 2, 2: 3, 9: 4, 4: 5, 3: 6, 10: 7, 11: 8, 12: 9, 5: 10, 6: 11, 8: 12, 7: 13,
        # 其他 Flare 映射
    },
    'word': {
        1: 1, 3: 2, 4: 3, 6: 4, 8: 5, 2: 6, 7: 7, 5: 8, 9: 9,
        # 其他 WORD 映射
        14: 14, 10: 16, 11: 17, 12: 18, 13: 19, 15: 20, 16: 21
    },
    'amos': {
        6: 1, 3: 2, 2: 3, 4: 4, 10: 5, 1: 6, 5: 7, 7: 8, 13: 9, 8: 10, 9: 11, 12: 12, 11: 13,
        # 其他 AMOS 映射
        14: 14, 15: 15
    }
}


def remap_labels(label_slice, mapping):
    """重新映射索引为1的标签层面"""
    remapped_labels = np.copy(label_slice)
    for original_label, new_label in mapping.items():
        remapped_labels[label_slice == original_label] = new_label
    return remapped_labels


def process_files(source_folder, target_folder, dataset_name):
    """处理文件夹中的所有npz文件并保存到新路径"""
    mapping = label_mappings[dataset_name]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 使用tqdm创建进度条
    file_list = os.listdir(source_folder)
    for file_name in tqdm(file_list, desc=f"处理 {dataset_name}"):
        if file_name.endswith('.npz'):
            file_path = os.path.join(source_folder, file_name)
            with np.load(file_path) as data:
                # 假设数据数组的名称是'data'
                images = data['data'][0, :, :]  # 索引为0的层面（不处理）
                labels = data['data'][1, :, :]  # 索引为1的层面（标签）
                remapped_labels = remap_labels(labels, mapping)
                # 将处理后的标签与未处理的图像层面组合回去
                processed_data = np.array([images, remapped_labels])

                # 保存到新的npz文件
                new_file_name = os.path.splitext(file_name)[0] + '_remapped.npz'
                new_file_path = os.path.join(target_folder, new_file_name)
                np.savez_compressed(new_file_path, data=processed_data)


def compare_data(source_folder, target_folder, dataset_name, report_file='mismatch_report.txt'):
    """比较原始文件和映射后文件的图像和标签数据是否相等，并记录结果"""
    mapping = label_mappings[dataset_name]
    all_equal = True
    files_to_check = [f for f in os.listdir(source_folder) if f.endswith('.npz')]

    with open(report_file, 'w') as report:
        for file_name in tqdm(files_to_check, desc="Comparing files"):
            # 加载原始文件
            original_file_path = os.path.join(source_folder, file_name)
            with np.load(original_file_path) as original_data:
                original_images = original_data['data'][0, :, :]
                original_labels = original_data['data'][1, :, :]

            # 加载映射后的文件
            remapped_file_name = os.path.splitext(file_name)[0] + '_remapped.npz'
            remapped_file_path = os.path.join(target_folder, remapped_file_name)
            with np.load(remapped_file_path) as remapped_data:
                remapped_images = remapped_data['data'][0, :, :]
                remapped_labels = remapped_data['data'][1, :, :]

            # 比较图像层面的数据是否相等
            if not np.array_equal(original_images, remapped_images):
                report.write(f"图像数据不匹配：{file_name}\n")
                all_equal = False

            # 比较标签层面的数据是否与原始映射相等
            if not np.array_equal(remap_labels(original_labels, mapping), remapped_labels):
                report.write(f"标签数据不匹配：{file_name}\n")
                all_equal = False

            if all_equal:
                report.write(f"{remapped_file_name} 文件匹配\n")
            else:
                all_equal = True  # 重置标志位为下一个文件检查

    if all_equal:
        print("所有文件的图像和标签数据在映射后保持一致。")
    else:
        print("有文件的图像和标签数据在映射后不一致，请查看报告。")


if __name__ == "__main__":
    # 遍历路径的字典
    dataset_paths = {
        'flare': (r"D:\Research\Swin-Unet\datasets\data\val\val_slices_flare", './remap/flare_val'),
        'word': (r"D:\Research\Swin-Unet\datasets\data\val\val_slices_word", './remap/word_val'),
        'amos': (r"D:\Research\Swin-Unet\datasets\data\val\val_slices_amos", './remap/amos_val'),
    }

    # 处理每个数据集
    for dataset_name, (source_path, target_path) in dataset_paths.items():
        process_files(source_path, target_path, dataset_name)
        print(f"处理完成: {dataset_name}")

        # 比较处理前后的数据集
        compare_data(source_path, target_path, dataset_name)
        print(f"数据比较完成: {dataset_name}")