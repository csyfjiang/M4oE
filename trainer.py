import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from miseval import evaluate
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):

    batch = [b for b in batch if b is not None]

    if len(batch) == 0: return None

    return default_collate(batch)


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    transforms_list = [
        RandomGenerator(output_size=[args.img_size, args.img_size]),
        # NormalizeSlice(),
    ]

    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(
        csv_file=args.data_csv,  # Assuming there is a csv file for training data
        transform=transforms.Compose(transforms_list),
        modes='train'
    )

    db_val = Synapse_dataset(
        csv_file=args.val_data_csv,  # Assuming there is a csv file for training data
        transform=transforms.Compose(transforms_list),
        modes='val'
    )
    trainloader = DataLoader(db_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True,
                             collate_fn=custom_collate_fn,
                             )

    valloader = DataLoader(db_val,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=8,
                           pin_memory=True,
                           collate_fn=custom_collate_fn,
                           )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")
    device = torch.device("cuda")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ce_loss_functions = {}
    for i, n in enumerate(num_classes):
        ce_loss_functions[str(n)] = nn.CrossEntropyLoss(weight=torch.ones(n).to(device), ignore_index=-1)


    dice_loss_functions = {}
    for i, n in enumerate(num_classes):
        dice_loss_functions[str(n)] = DiceLoss(n_classes=n)

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # early stopping
    val_loss_min = np.Inf
    patience = 7
    patience_counter = 0

    # train loop
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            if sampled_batch is None:
                continue
            image_batch, label_batch, dataset_id, predict_head, n_classes = sampled_batch['image'], sampled_batch[
                'label'], \
                sampled_batch['dataset_id'], sampled_batch['predict_head'], sampled_batch['n_classes']

            # Ensure dataset_id and predict_head are tensors and send to device
            if not isinstance(dataset_id, torch.Tensor):
                dataset_id = torch.tensor(dataset_id)
            dataset_id = dataset_id.to(device)

            if not isinstance(predict_head, torch.Tensor):
                predict_head = torch.tensor(predict_head)
            predict_head = predict_head.to(device)
            loss_ce = 0
            loss_dice = 0
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            # pass dataset_id and predict_head to model
            outputs = model(image_batch, dataset_id, predict_head)

            for i in range(image_batch.size(0)):
                num_classes_i = str(n_classes[i].item())
                ce_loss_func = ce_loss_functions[num_classes_i]
                dice_loss_func = dice_loss_functions[num_classes_i]
                output_i = outputs[i, :n_classes[i].item()].unsqueeze(0)

                labels_i = label_batch[i].long().unsqueeze(0)
                # print(labels_i.unique(), n_classes[i].item())
                loss_ce_i = ce_loss_func(output_i, labels_i)

                # print(labels_i.unique(),n_classes[i].item())

                loss_dice_i = dice_loss_func(output_i, labels_i, softmax=True)

                loss_ce += loss_ce_i
                loss_dice += loss_dice_i

            loss_ce /= len(predict_head)
            loss_dice /= len(predict_head)


            loss = 0.4 * loss_ce + 0.7 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # Write  log information to tensorboard
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f Dice: %f' % (
                iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                if image_batch.size(0) > 1:
                    image = image_batch[1, 0:1, :, :]
                else:
                    print("Warning: image_batch only contains one image.")
                    image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

                if outputs.size(0) > 1:
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                else:
                    print("Warning: outputs only contains one prediction.")
                    writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)

                if label_batch.size(0) > 1:
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                else:
                    print("Warning: label_batch only contains one label.")
                    labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # validation loop
        if epoch_num % 2 == 0:
            model.eval()
            with torch.no_grad():
                val_loss_ce = 0
                val_loss_dice = 0
                for i_batch, sampled_batch in enumerate(valloader):
                    if sampled_batch is None:
                        continue
                    image_batch, label_batch, dataset_id, predict_head, n_classes = sampled_batch['image'], \
                    sampled_batch[
                        'label'], \
                        sampled_batch['dataset_id'], sampled_batch['predict_head'], sampled_batch['n_classes']

                    # Ensure dataset_id and predict_head are tensors and send to device
                    if not isinstance(dataset_id, torch.Tensor):
                        dataset_id = torch.tensor(dataset_id)
                    dataset_id = dataset_id.to(device)

                    if not isinstance(predict_head, torch.Tensor):
                        predict_head = torch.tensor(predict_head)
                    predict_head = predict_head.to(device)

                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                    # pass dataset_id and predict_head to model
                    val_outputs = model(image_batch, dataset_id, predict_head)

                    for i in range(image_batch.size(0)):
                        num_classes_i = str(n_classes[i].item())
                        val_ce_loss_func = ce_loss_functions[num_classes_i]
                        val_dice_loss_func = dice_loss_functions[num_classes_i]

                        output_i = val_outputs[i, :n_classes[i].item()].unsqueeze(0)  # 确保输出与类别数匹配

                        labels_i = label_batch[i].long().unsqueeze(0)

                        # CrossEntropyLoss
                        val_loss_ce_i = val_ce_loss_func(output_i, labels_i)
                        # DiceLoss
                        val_loss_dice_i = val_dice_loss_func(output_i, labels_i, softmax=True)

                        val_loss_ce += val_loss_ce_i
                        val_loss_dice += val_loss_dice_i

                    val_loss_ce /= len(predict_head)
                    val_loss_dice /= len(predict_head)

                    val_loss = 0.4 * val_loss_ce + 0.6 * val_loss_dice

                    writer.add_scalar('info/val_loss', val_loss, iter_num)
                    writer.add_scalar('info/val_loss_ce', val_loss_ce, iter_num)
                    writer.add_scalar('info/val_loss_dice', val_loss_dice, iter_num)

                    model.train()
                    # Check for improvement

                if val_loss < val_loss_min:
                    logging.info(f'Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
                    # Save the model if validation loss has decreased
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    val_loss_min = val_loss
                    patience_counter = 0  # Reset patience counter
                else:
                    patience_counter += 1
                    logging.info(f'EarlyStopping counter: {patience_counter} out of {patience}')
                    if patience_counter >= patience:
                        logging.info('Early stopping triggered. Stopping training...')
                        iterator.close()
                        break  # Early stopping

        # save_interval = 25  # int(max_epoch/6)
        # # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        # if (epoch_num+1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or patience_counter >= patience:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == '__main__':
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    db_train = Synapse_dataset(
        csv_file=r"./lists/flare.csv",  # Assuming there is a csv file for training data
        transform=transforms.Compose([RandomGenerator(output_size=[224, 224]), ]),
        modes='train'
    )
    print("The length of train set is: {}".format(len(db_train)))
    trainloader = DataLoader(db_train,
                             batch_size=12,
                             shuffle=True,
                             num_workers=6,
                             pin_memory=True,
                             worker_init_fn=None,
                             collate_fn=custom_collate_fn)

    for i, data in enumerate(trainloader):
        image_batch, label_batch, dataset_id, predict_head = data['image'], data['label'], data['dataset_id'], data[
            'predict_head']
        print("Type of image_batch:{}".format(type(image_batch)))
        print("image_batch size:{}".format(image_batch.size()))
        print("label_batch size:{}".format(label_batch.size()))
        print("dataset_id size:{}".format(dataset_id.size()))
        print("dataset_id:{}".format(dataset_id))
        print("predict_head:{}".format(predict_head))
        print("num_classes:{}".format(data['n_classes']))
        print('------------------')
        if i == 3:
            break
