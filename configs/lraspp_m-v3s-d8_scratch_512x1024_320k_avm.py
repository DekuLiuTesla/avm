_base_ = ["lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes.py"]

NUM_CLASSES = 4
PRETRAINED = "lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes_20201224_223935-03daeabb.pth"

# dataset settings
dataset_type = 'AVMDataset'
data_root = "avm"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='semantic',
        split = "train.txt",),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='semantic',
        split = "test.txt",),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='semantic',
        split = "test.txt",))

model = dict(
    pretrained=PRETRAINED,
    decode_head=dict(num_classes=NUM_CLASSES)
)



