import os
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import numpy as np

def nii_to_png(image_path, label_path, output_dir, dsname):
    img = nib.load(image_path).get_fdata()
    lbl = nib.load(label_path).get_fdata()
    image_base_name = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]

    if dsname == 'Flare22':
        label_base_name = image_base_name.replace('_0000', '')
    elif dsname == 'WORD':
        label_base_name = image_base_name
    elif dsname == 'BTCV':
        label_base_name = image_base_name.replace('.nii.gz', '_seg.nii.gz')
    elif dsname == 'AMOS':
        label_base_name = image_base_name

    os.makedirs(os.path.join(output_dir, 'images', image_base_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', label_base_name), exist_ok=True)

    saved_slices = 0

    for i in tqdm(range(img.shape[2]), desc="Converting slices"):
        _, counts = np.unique(lbl[:, :, i], return_counts=True)
        if (1 - (counts[0] / counts.sum())) > 0.01:
            image_slice = Image.fromarray((img[:, :, i] * 255).astype(np.uint8))
            image_slice.save(os.path.join(output_dir, 'images', image_base_name, f'{image_base_name}_slice_{saved_slices}.png'))

            label_slice = Image.fromarray(lbl[:, :, i].astype(np.uint8))
            label_slice.save(os.path.join(output_dir, 'labels', label_base_name, f'{label_base_name}_slice_{saved_slices}.png'))

            saved_slices += 1

datasets = [
    {
        'name': 'Flare22',
        'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\FLARE22\Train\Training-20230908T070644Z-002\Training\FLARE22_LabeledCase50\images',
        'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\FLARE22\Train\Training-20230908T070644Z-002\Training\FLARE22_LabeledCase50\labels'
    },
    {
        'name': 'WORD',
        'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\WORD-V0.1.0\WORD-V0.1.0\imagesTr',
        'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\WORD-V0.1.0\WORD-V0.1.0\labelsTr'
    },
    {
        'name': 'BTCV',
        'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\BTCV\averaged-training-images',
        'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\BTCV\averaged-training-labels'
    },
    {
        'name': 'AMOS',
        'image_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\amos22\amos22\imagesTr',
        'label_dir': r'C:\Users\90515\Desktop\Research\FoundationModel\datasets\amos22\amos22\labelsTr'
    }
]


for dataset in datasets:
    dsname = dataset['name']
    image_dir = dataset['image_dir']
    label_dir = dataset['label_dir']
    output_dir = r'D:\Research\Swin-Unet\imgdata\{}'.format(dsname.lower())

    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith('.nii.gz'):
            continue

        if dsname == 'Flare22':
            label_filename = image_filename.replace('_0000', '')
        elif dsname == 'WORD':
            label_filename = image_filename
        elif dsname == 'BTCV':
            label_filename = image_filename.replace('.nii.gz', '_seg.nii.gz')
        elif dsname == 'AMOS':
            label_filename = image_filename

        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, label_filename)

        nii_to_png(image_path, label_path, output_dir, dsname)