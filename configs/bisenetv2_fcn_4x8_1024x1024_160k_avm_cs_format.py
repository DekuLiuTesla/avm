_base_ = [
    'bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py'
]
data_root = 'data/avm_cs_format/'

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train'),
    val=dict(
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train'),
    test=dict(
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train'))
