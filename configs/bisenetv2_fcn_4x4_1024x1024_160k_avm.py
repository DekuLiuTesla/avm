_base_ = ["bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py"]

NUM_CLASSES = 4
PRETRAINED = "bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth"

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
    decode_head=dict(num_classes=NUM_CLASSES),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=PRETRAINED,
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=NUM_CLASSES,
            in_index=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=NUM_CLASSES,
            in_index=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=NUM_CLASSES,
            in_index=3,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=NUM_CLASSES,
            in_index=4,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],

)

