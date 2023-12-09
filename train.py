from argparse import ArgumentParser

import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss
from tqdm import trange, tqdm

from Dataloader import kitti_data_loader, vkitti2_data_loader
from Models import MiDaS


def train(args):
    if args.dataset == 'kitti_large':
        train_loader = kitti_data_loader.get_kitti_loader(
            'Dataloader/kitti', split='train-large',
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = kitti_data_loader.get_kitti_loader(
            'Dataloader/kitti', split='test',
            batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'kitti_small':
        train_loader = kitti_data_loader.get_kitti_loader(
            'Dataloader/kitti', split='train-small',
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = kitti_data_loader.get_kitti_loader(
            'Dataloader/kitti', split='test',
            batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'vkitti2':
        train_loader = vkitti2_data_loader.get_vkitti2_loader(
            'Dataloader/vkitti2', split='train',
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = vkitti2_data_loader.get_vkitti2_loader(
            'Dataloader/vkitti2', split='test',
            batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise NotImplementedError

    net = MiDaS.get_model(args.model_type)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    criterion = MSELoss()

    for epoch in trange(args.epoch, ncols=65):
        net.train()
        train_loss = []
        for i, sample in enumerate(tqdm(train_loader)):

            inputs = sample['image'].to(args.device)
            targets = sample['depth'].to(args.device)
            outputs = net(inputs)
            loss = criterion(targets, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm.set_description(
                f'Train|epoch: {epoch}|step: {i}|loss: {loss.to("cpu").tolist()}')
            train_loss.append(loss.to('cpu').tolist())
        tqdm.write(f'Train|epoch: {epoch}|loss {np.mean(train_loss)}')

        if epoch % args.eval_epoch == 0:
            test_loss = []
            with torch.no_grad():
                net.eval()
                for i, sample in enumerate(test_loader):
                    inputs = sample['image'].to(args.device)
                    targets = sample['depth'].to(args.device)
                    outputs = net(inputs)
                    loss = criterion(targets, outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tqdm.set_description(
                        f'Test|epoch: {epoch}|step: {i}|loss: {loss.to("cpu").tolist()}')
                    test_loss.append(loss.to('cpu').tolist())
                tqdm.write(f'Test|epoch: {epoch}|loss: {np.mean(test_loss)}')
            


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='MiDaS_small')

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args=args)
