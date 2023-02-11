_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.1)
runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(by_epoch=False, interval=1000)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val=dict(
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train'),
    test=dict(
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train')
)
