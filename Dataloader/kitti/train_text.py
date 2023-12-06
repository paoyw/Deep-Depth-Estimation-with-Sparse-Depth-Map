import glob

#depth/train/*/proj_depth/groundtruth/image_02/0000000005.png
#2011_09_26/*/image_02/data/0000000005.png

base_path = 'depth/train/'
files = glob.glob(base_path + '2011_09_26*/proj_depth/groundtruth/image_02/*.png')
files.sort()


# 寫入到一個 txt 檔案中

output_file = './train.txt'
with open(output_file, 'w') as f:
    for file in files:
        f.write(file + '\n')







