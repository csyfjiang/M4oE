import numpy as np
import os
from tqdm import tqdm

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


def remap_labels(label_volume, mapping):
    """重新映射整个标签体数据"""
    # 创建一个与标签体数据形状相同的数组，用于保存重新映射的标签
    remapped_labels = np.copy(label_volume)
    for original_label, new_label in mapping.items():
        remapped_labels[label_volume == original_label] = new_label
    return remapped_labels


def process_files(source_folder, target_folder, dataset_name):
    """处理文件夹中的所有npz文件并保存到新路径"""
    mapping = label_mappings[dataset_name]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 使用tqdm创建进度条
    file_list = os.listdir(source_folder)
    processed_count = 0
    for file_name in tqdm(file_list, desc=f"处理 {dataset_name}"):
        try:
            if file_name.endswith('.npz'):
                file_path = os.path.join(source_folder, file_name)
                with np.load(file_path) as data:
                    images = data['data'][0, :, :, :]
                    labels = data['data'][1, :, :, :]
                    remapped_labels = remap_labels(labels, mapping)
                    processed_data = np.array([images, remapped_labels])

                    new_file_name = os.path.splitext(file_name)[0] + '_remapped.npz'
                    new_file_path = os.path.join(target_folder, new_file_name)
                    np.savez_compressed(new_file_path, data=processed_data)
                    processed_count += 1
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

    print(f"成功处理并保存了 {processed_count} 个文件到 {target_folder}.")

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
                original_images = original_data['data'][0, :, :, :]
                original_labels = original_data['data'][1, :, :, :]

            # 加载映射后的文件
            remapped_file_name = os.path.splitext(file_name)[0] + '_remapped.npz'
            remapped_file_path = os.path.join(target_folder, remapped_file_name)
            with np.load(remapped_file_path) as remapped_data:
                remapped_images = remapped_data['data'][0, :, :, :]
                remapped_labels = remapped_data['data'][1, :, :, :]

            # 比较图像层面的数据是否相等
            if not np.array_equal(original_images, remapped_images):
                report.write(f"图像数据不匹配：{file_name}\n")
                all_equal = False

            # 比较标签层面的数据是否与原始映射相等
            if not np.array_equal(remap_labels(original_labels, mapping), remapped_labels):
                report.write(f"标签数据不匹配：{file_name}\n")
                all_equal = False

            if not all_equal:
                report.write(f"{file_name} 文件存在不匹配项\n")
                all_equal = True  # 重置标志位为下一个文件检查

    if all_equal:
        print("所有文件的图像和标签数据在映射后保持一致。")
    else:
        print("有文件的图像和标签数据在映射后不一致，请查看报告。")


# 确保 remap_labels 和 label_mappings 与之前定义的一致

# 使用该函数
if __name__ == "__main__":
    # label_mappings 定义应存在于此处

    dataset_paths = {
        'flare': (r"J:\datasets\testing\Task204_FLARE22Test\nnUNetData_plans_v2.1_stage0", r"J:\datasets\testing\remap_test\remap_flare"),
        'word': (r"J:\datasets\testing\Task203_WORDTest\nnUNetData_plans_v2.1_stage0", r"J:\datasets\testing\remap_test\remap_word"),
        'amos': (r"J:\datasets\testing\Task202_AMOSTest\nnUNetData_plans_v2.1_stage0", r"J:\datasets\testing\remap_test\remap_amos"),
    }

    for dataset_name, (source_path, target_path) in dataset_paths.items():
        process_files(source_path, target_path, dataset_name)  # 确保 process_files 函数已正确定义
        print(f"处理完成: {dataset_name}")
        compare_data(source_path, target_path, dataset_name)
        print(f"数据比较完成: {dataset_name}")