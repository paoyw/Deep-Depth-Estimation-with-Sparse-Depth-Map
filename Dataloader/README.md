# Dataloader

## Dataset Size (one camera)
### KITTI:

Training(Large): 13327 images

Training(Small): 1709 images

Testing: 2103 images

### VKITTI2:
Training: 19959 images

Testing: 1701 images

#### Traing with two cameras images, the size will be double 

## Usage

```python
#vkitti2
from vkitti2_data_loader import get_vkitti2_loader
loader = get_vkitti2_loader( data_dir_root="./vkitti2", split='train') #for training
loader = get_vkitti2_loader( data_dir_root="./vkitti2", split='test') #for testing

#kitti
from kitti_data_loader import get_kitti_loader
loader = get_kitti_loader( data_dir_root="./kitti", split='train') #for training
loader = get_kitti_loader( data_dir_root="./kitti", split='test') #for testing

for i, sample in enumerate(loader):
        print(sample["image"].shape) #camera0 image: torch.Size([1, 3, 375, 1242])
        print(sample["depth"].shape) #camera0 depth: torch.Size([1, 1, 375, 1242])
        print(sample["image2"].shape) #camera1 image torch.Size([1, 3, 375, 1242])
        print(sample["depth2"].shape) #camera1 depth torch.Size([1, 1, 375, 1242])
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max()) #if depth>80 adjust it to-1 in preprocess

```

## File path


```
└── root
    ├── kitti/
    ├── vkitti2/
    ├── kitti_data_loader.py
    └── vkitti2_data_loader.py
```

