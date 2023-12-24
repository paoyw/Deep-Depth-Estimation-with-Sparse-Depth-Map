# NTU 3D Computer Vision Final Project


## Data Downloads
Please see the `README.md` under `DataLoader`


## Usage

## How to train the model

```python
python train.py --data kitti_large --ckpt Models/ckpt/autoencoder_kitti_large.pt
python train.py --data kitti_small --ckpt Models/ckpt/autoencoder_kitti_small.pt
python train.py --data vkitti2 --ckpt Models/ckpt/autoencoder_vkitti2.pt
```

## How to test the model

```python
python test.py --dataset kitti_large --load_ckpt Models/ckpt/autoencoder_vkitti2.pt
python test.py --dataset kitti_large --load_ckpt Models/ckpt/autoencoder_kitti_large.pt
python test.py --dataset kitti_large --load_ckpt Models/ckpt/autoencoder_kitti_small.pt
python test.py --dataset kitti_large --load_ckpt Models/ckpt/autoencoder_kitti_tune.pt
```
## How to inference the model
```python
python inference.py \
    --ckpt [model weight path] \
    --mode ['q3' | 'q1_q3' | 'mean'] \
    --input_img [imput image path] \
    --ground_trute [ground truth depth path] \
    --icp [store true to turn on icp]
```
example:

```python
python inference.py \
    --ckpt Models/ckpt/autoencoder_vkitti2.pt \
    --mode q3 \
    --input_img Dataloader/kitti/RGB/0000000005.png \
    --ground_truth Dataloader/kitti/Depth/0000000005.png
```
## File path

```
└── root
    ├── data_for_test
    │   ├── depth.png
    │   └── rgb.png
    ├── Dataloader/
        ├── kitti/
        ├── vkitti2/
        ├── kitti_data_loader.py
        └── vkitti2_data_loader.py
    ├── Models/
        ├── ckpt/
        ├── __init__.py
        ├── AutoEncoder.py
        └── Loss.py
    ├── depthadjust.py
    ├── exp_plot.ipynb
    ├── inference.py
    ├── README.md
    ├── train.py
    └── test.py
    
```