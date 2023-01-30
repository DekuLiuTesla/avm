# AVM 

## 训练

在当前目录下建立软连接
```bash
ln -s AVM_DATA_ROOT avm
```

其中AVM原始数据集目录如下
```python
avm
 │
 ├─ annotations
 │   ├─ 00000001.yml
 │   └─ 00000002.yml
 │       ... 
 ├─ gt
 │   ├─ 00000001.png
 │   └─ 00000002.png
 │
 ├─ images
 │   ├─ 00000001.jpg
 │   └─ 00000002.jpg
 │
 ├─ info.txt
 │
 ├─ test_db.txt
 │
 ├─ train_db.txt
 │
 ├─ show_db_info.py
 │
 └─ split_db.py
```

将原数据集格式转换为VOC格式
```python
python convert_avm2voc.py --root=avm --nproc=10
```

转换过后文件结构如下：
```python
avm
 │
 ├─ annotations
 │   ├─ 00000001.yml
 │   └─ 00000002.yml
 │       ... 
 ├─ gt
 │   ├─ 00000001.png
 │   └─ 00000002.png
 │
 ├─ images
 │   ├─ 00000001.jpg
 │   └─ 00000002.jpg
 │
 │   ... 
 │
 ├─ train.txt
 │
 ├─ test.txt
 │
 └─ semantic
     ├─ 00000001.png
     └─ 00000002.png
 
```

将avm.py文件复制到mmsegmentation/mmseg/datasets/下
并修改mmsegmentation/mmseg/datasets/\_\_init\_\_.py

```python
...
from .avm import AVMDataset

__all__ = [..., 'AVMDataset']
```

## 推理

```bash
python inference.py --config=upernet_swin_small_patch4_window7_512x512_160k_avm20k_pretrain_224x224_1K.py --checkpoint=latest.pth --image_path=demo.jpg 
```
