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

## File path

```
└── root
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
    ├── README.md
    ├── train.py
    └── test.py
    
```