import os
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import numpy as np

def nii_to_png(image_path, output_dir, dsname):
    img = nib.load(image_path).get_fdata()
    image_base_name = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]

    os.makedirs(os.path.join(output_dir, 'image'), exist_ok=True)

    saved_slices = 0

    for i in tqdm(range(img.shape[2]), desc="Converting slic es"):
        # image_slice = Image.fromarray((img[:, :, i] * 255).astype(np.uint8))
        # image_slice.save(os.path.join(output_dir, image_base_name, f'{image_base_name}_slice_{saved_slices}.png'))
        # saved_slices += 1
        image_slice = Image.fromarray((img[:, :, i] * 255).astype(np.uint8))
        image_slice_filename = f'{image_base_name}_slice_{saved_slices}.png'
        image_slice.save(os.path.join(output_dir,'image', image_slice_filename))
        saved_slices += 1

datasets = [
    # {
    #     'name': 'Flare22',
    #     'image_dir': r"J:\datasets\medical\dataset\nnUNet_raw_data_base\nnUNet_raw_data\Task201_FLARE22labeled\imagesTr",
    # },
    # {
    #     'name': 'WORD',
    #     'image_dir': r"J:\datasets\medical\dataset\nnUNet_raw_data_base\nnUNet_raw_data\Task203_WORD\imagesTr",
    # },
    # {
    #     'name': 'AMOS',
    #     'image_dir': r"J:\datasets\medical\dataset\nnUNet_raw_data_base\nnUNet_raw_data\Task202_AMOS\imagesTr",
    # },
    {
        'name': 'ALTAS',
        'image_dir': r"J:\datasets\medical\dataset\nnUNet_raw_data_base\nnUNet_raw_data\Task205_ALTAS\imagesTr",
    },
    {
        'name': 'AMOS_MRclear',
        'image_dir': r"J:\datasets\medical\dataset\nnUNet_raw_data_base\nnUNet_raw_data\Task219_AMOS2022_postChallenge_MR\imagesTr",
    }
]


if __name__ == '__main__':
    for dataset in datasets:
        dsname = dataset['name']
        image_dir = dataset['image_dir']
        output_dir = r"J:\datasets\medical\self_supervised_source_png" # 50920
        os.makedirs(output_dir, exist_ok=True)
        for image_filename in os.listdir(image_dir):
            if not image_filename.endswith('.nii.gz'):
                continue

            image_path = os.path.join(image_dir, image_filename)
            nii_to_png(image_path, output_dir, dsname)