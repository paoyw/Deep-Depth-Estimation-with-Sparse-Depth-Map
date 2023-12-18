from argparse import ArgumentParser

import numpy as np
import torch
from torch import optim
from tqdm import trange, tqdm

from Dataloader import kitti_data_loader, vkitti2_data_loader
from Models import AutoEncoder, Loss


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

    net = AutoEncoder.AutoEncoder().to(args.device)

    if args.load_ckpt is not None:
        state_dict = torch.load(args.load_ckpt)
        net.load_state_dict(state_dict)
        net = net.to(args.device)

    optimizer = optim.Adam(net.decoder.parameters(), lr=args.lr)

    criterion = Loss.MaskedMSE()

    for epoch in trange(args.epoch, ncols=100):
        net.train()
        train_loss = []
        pbar = tqdm(train_loader, ncols=100)
        optimizer.zero_grad()
        for i, sample in enumerate(pbar):
            inputs = sample['image'].to(args.device)
            targets = sample['depth'].to(args.device)
            
            outputs = net(inputs)
            
            loss = criterion(outputs, targets / 80, targets >= 0)

            loss.backward()
            if i % args.gradient_accumulate == args.gradient_accumulate - 1 or i == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description_str(
                f'Train|epoch: {epoch}|step: {i}|loss: {loss.to("cpu").tolist():.5f}|mean: {outputs.mean().to("cpu").tolist():.3f}')
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
                    loss = criterion(outputs, targets / 80, targets >= 0)
                    test_loss.append(loss.to('cpu').tolist())
                tqdm.write(f'Test|epoch: {epoch}|loss: {np.mean(test_loss)}')
    
    state_dict = net.to('cpu').state_dict()
    torch.save(state_dict, args.ckpt)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='autoencoder')
    parser.add_argument('--load_ckpt', type=str, default=None)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulate', type=int, default=8)

    parser.add_argument('--ckpt', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args=args)
