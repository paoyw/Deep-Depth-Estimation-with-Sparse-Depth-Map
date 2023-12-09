# kitti

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        self.normalize = lambda x: x

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image2, depth2= sample['image2'], sample['depth2']

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)
        image2 = self.to_tensor(image2)
        image2 = self.normalize(image2)
        depth2 = self.to_tensor(depth2)

        return {'image': image, 'depth': depth, 'image2': image2, 'depth2': depth2,'dataset': "kitti"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class kitti(Dataset):
    def __init__(self, data_dir_root, do_kb_crop=True, split="train"):
        import glob
        self.do_kb_crop = False
        self.transform = ToTensor()
        if split == "train-large":
            with open(os.path.join(data_dir_root, "train_13327.txt"), "r") as f:
                self.depth_files = f.read().splitlines()
            self.image_files=[]
            destination_base = './kitti/2011_09_26/{}/image_02/data/'
            for source_file in self.depth_files:
                parts = source_file.split('/')
                common_id = parts[2]  
                filename = parts[-1]  
                new_file_path = destination_base.format(common_id) + filename
                self.image_files.append(new_file_path)
            self.image_files2 = [r.replace("/image_02/", "/image_03/")for r in self.image_files]
            self.depth_files2 = [r.replace("/image_02/", "/image_03/")for r in self.depth_files]
        elif split == "train-small":
            with open(os.path.join(data_dir_root, "train_1709.txt"), "r") as f:
                self.depth_files = f.read().splitlines()
            self.image_files=[]
            destination_base = './kitti/2011_09_26/{}/image_02/data/'
            for source_file in self.depth_files:
                parts = source_file.split('/')
                common_id = parts[2]  
                filename = parts[-1]  
                new_file_path = destination_base.format(common_id) + filename
                self.image_files.append(new_file_path)
            self.image_files2 = [r.replace("/image_02/", "/image_03/")for r in self.image_files]
            self.depth_files2 = [r.replace("/image_02/", "/image_03/")for r in self.depth_files]
        elif split == "test":
            with open(os.path.join(data_dir_root, "test_2103.txt"), "r") as f:
                self.depth_files = f.read().splitlines()
            self.image_files=[]
            destination_base = './kitti/2011_09_26/{}/image_02/data/'
            for source_file in self.depth_files:
                parts = source_file.split('/')
                common_id = parts[2]  
                filename = parts[-1]  
                new_file_path = destination_base.format(common_id) + filename
                self.image_files.append(new_file_path)
            self.image_files2 = [r.replace("/image_02/", "/image_03/")for r in self.image_files]
            self.depth_files2 = [r.replace("/image_02/", "/image_03/")for r in self.depth_files]

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        image_path2 = self.image_files2[idx]
        depth_path2 = self.depth_files2[idx]

        image = Image.open(image_path)

        depth = cv2.imread(os.path.join('./kitti',depth_path), cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth = Image.fromarray(depth)

        image2 = Image.open(image_path2)

        depth2 = cv2.imread(os.path.join('./kitti',depth_path2), cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth2 = Image.fromarray(depth2)


        if self.do_kb_crop:
            if idx == 0:
                print("Using KB input crop")
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth[depth > 80] = -1

        depth = depth[..., None]

        image2 = np.asarray(image2, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth2 = np.asarray(depth2, dtype=np.float32) / 1.
        depth2[depth2 > 80] = -1

        depth2 = depth2[..., None]
        sample = dict(image=image, depth=depth,image2=image2, depth2=depth2)

        # return sample
        sample = self.transform(sample)

        #if idx == 0:
        #   print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_kitti_loader(data_dir_root, split='train', batch_size=1, **kwargs):
    dataset = kitti(data_dir_root,split=split)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_kitti_loader( data_dir_root="./kitti", split='train-small')
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):

        print(sample["image"].shape) #camera0 image
        '''
        image_tensor = sample["image"].squeeze(0).permute(1, 2, 0)  # Remove batch dimension and change to HWC format
        image_numpy = image_tensor.numpy()
        image_numpy = (image_numpy * 255).astype(np.uint8)  # Convert to uint8
        image_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        cv2.imshow('Image', image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        print(sample["depth"].shape) #camera0 depth
        '''
        image_tensor = sample["depth"].squeeze(0).permute(1, 2, 0)  # Remove batch dimension and change to HWC format
        image_numpy = image_tensor.numpy()
        image_numpy = (image_numpy *255).astype(np.uint8)  # Convert to uint8
        image_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        cv2.imshow('Image', image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        print(sample["image2"].shape) #camera1 image
        print(sample["depth2"].shape) #camera1 depth
        #print(sample["image"])
        #print(sample["image2"])
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break

    