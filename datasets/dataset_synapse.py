import logging
import os
import pandas as pd
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
import nibabel as nib
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, dataset_id, predict_head,n_classes = sample['image'], sample['label'], sample['dataset_id'], sample[
            'predict_head'], sample['n_classes']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        # image = image.permute(2, 0, 1)

        label = torch.from_numpy(label.astype(np.float32))

        # image = image.astype(np.float32)
        # label = label.astype(np.float32)
        sample = {'image': image, 'label': label, 'dataset_id': dataset_id, 'predict_head': predict_head, 'n_classes': n_classes}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, csv_file, transform=None, modes='train'):
        self.transform = transform
        self.dataframe = pd.read_csv(csv_file, sep=',')
        self.mode = modes

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        data_dir = row['data_dir']
        img_idx = row['img_idx']
        label_idx = row['label_idx']
        dataset_id = row['dataset_id']
        predict_head = row['predict_head']
        n_classes = row['n_classes']


        npz_data = np.load(data_dir)
        data = npz_data['data']


        if self.mode == 'train':

            image = data[img_idx]
            label = data[label_idx]

        else:

            image = data[img_idx]
            label = data[label_idx]

        filename = os.path.basename(data_dir)

        case_name = filename.split('.')[0]

        sample = {
            'image': image,
            'label': label,
            'dataset_id': dataset_id,
            'predict_head': predict_head,
            'n_classes': n_classes,
            'case_name': case_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    csv_file = '../lists/datasets_v10.csv'
    import numpy as np
    from PIL import Image

    transforms_list = [

        RandomGenerator(output_size=[224, 224]),
        # NormalizeSlice(),
        # Custom transformation
    ]

    # max_iterations = args.max_iterations
    dataset = Synapse_dataset(
        csv_file=csv_file,# Assuming there is a csv file for training data
        transform=transforms.Compose(transforms_list),
        modes='train'
    )

    # Print the dataset length
    print(f'Dataset length: {len(dataset)}')
    # print(dataset[0])
    # Print out the information for the samples in the specified range
    for i in range(1):
        sample = dataset[i]
        image = sample['image']
        label = sample['label']  # Assuming this is the mask
        print(sample)
        # case_name = sample['case_name']
        print(type(image))
        print(image.shape)
        # print(case_name)
        # print(f"Sample {i}:")
        # # print(iou)
        # print(image.shape)
        # print(label.shape)
        # print(f"Image range: min={np.min(np.array(image))}, max={np.max(np.array(image))}")
        # # print(f"Image range: min={np.min(image)}, max={np.max(image)}")
        # # print(f"Label (mask) range: min={np.min(label)}, max={np.max(label)}")
        # print("--------------------------------------------------")
