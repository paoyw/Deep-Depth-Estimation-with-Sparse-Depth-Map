from argparse import ArgumentParser
from PIL import Image
import torch
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

from Models.AutoEncoder import AutoEncoder
from depthadjust import DepthAdjust

def read_data(img_path, depth_path):
    
    rgb_image = Image.open(img_path)
    height = rgb_image.height
    width = rgb_image.width
    top_margin = int(height - 352)
    left_margin = int((width - 1216) / 2)
    image = rgb_image.crop(
        (left_margin, top_margin, left_margin + 1216, top_margin + 352))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image.transpose((2,0,1)))
    image = image.unsqueeze(0)
    
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0
    depth = Image.fromarray(depth)
    depth = depth.crop(
        (left_margin, top_margin, left_margin + 1216, top_margin + 352))
    depth = np.asarray(depth, dtype=np.float32) / 1.
    depth[depth > 80] = -1

    depth = depth[..., None]
    depth = torch.from_numpy(depth.transpose((2, 0, 1)))
    depth = depth.unsqueeze(0)
    return rgb_image, image, depth
    
def main(args):
    model = AutoEncoder()
    with open(args.ckpt, 'rb') as f:
        weights = torch.load(f)
        model.load_state_dict(weights)
    model = model.cuda().eval()

    dj = DepthAdjust()

    rgb_image, image, depth = read_data(args.input_img, args.ground_truth)
    
    
    
    with torch.no_grad():
        pred = model(image.cuda()).cpu()

    pred_for_adj = torch.permute(pred, (2,3,1,0)).squeeze(-1).numpy()
    depth_for_adj = torch.permute(depth, (2,3,1,0)).squeeze(-1).numpy()
    
    _,_,pred = dj.fit_and_trainsform(pred_for_adj, depth_for_adj, mode=args.mode, icp=args.icp, return_proj=True)

    
    fig, ax = plt.subplots(3,1)
    print(ax)
    ax[0].set_title('rgb image')
    ax[0].axis("off")
    ax[0].imshow(rgb_image)
    ax[1].set_title('GT depth')
    ax[1].axis('off')
    ax[1].imshow(depth.squeeze())
    ax[2].set_title('predict depth')
    ax[2].axis('off')
    ax[2].imshow(pred)
    plt.show()
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--ckpt',
        type=str,
    )
    parser.add_argument(
        '--icp',
        action='store_true'
    )
    parser.add_argument(
        '--mode',
        type=str
    )
    parser.add_argument(
        '--input_img',
        type=str
    )
    parser.add_argument(
        '--ground_truth',
        type=str
    )
    args = parser.parse_args()
    main(args)