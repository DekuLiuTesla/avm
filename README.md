# AVM 

## 环境

### 最佳实践（推荐，已测试可行）

1. 第一步 使用 MIM 安装 MMCV
```bash
pip install -U openmim
mim install mmcv-full
```

2. 第二步 安装 MMSegmentation
```bash
pip install mmsegmentation
```

3. 第三步下载配置文件(可选)

```bash
mim download mmsegmentation --config bisenetv2_fcn_4x4_1024x1024_160k_cityscapes --dest .
```

## 训练

在当前目录下建立软连接
```bash
ln -s AVM_DATA_ROOT avm
```

其中AVM原始数据集目录需调整为如下结构
```python
avm
 │
 ├─ images
 │   ├─ b2_left
 │   │   └─ avm
 │   │       ├─ 0.jpg
 │   │       └─ 20.jpg
 │   │       ... 
 │   ├─ b2_right
 │   ├─ b2_to_b3
 │   └─ b3_to_b2
 │   
 └─ mask
     ├─ b2_left
     │   ├─ 0.jpg
     │   └─ 20.jpg
     │   ... 
     ├─ b2_right
     ├─ b2_to_b3
     └─ b3_to_b2
```

将原数据集格式转换为CityScapes格式
```python
python create_data.py --avm-dir ./data/avm --output-dir ./data/avm_cs_format
```

转换过后文件结构如下：
```python
avm_cs_format
 │
 ├─ gtFine
 │   └─ train
 │       ├─ b2_left_0_gtFine_labelTrainIds.png
 │       └─ b2_left_1000_gtFine_labelTrainIds.png
 │       ... 
 └─ leftImg8bit
     └─ train
         ├─ b2_left_0_leftImg8bit.png
         └─ b2_left_1000_leftImg8bit.png
         ... 
 
```


## 推理

```bash
python inference.py --config=configs/bisenetv2_fcn_4x8_1024x1024_160k_avm_cs_format.py --checkpoint=iter_5000.pth --image_path=demo.jpg 
```
