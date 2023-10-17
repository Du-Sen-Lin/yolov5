# Yolov5

fork by 2023.10.16

## 一、实验测试

```python
# 环境安装和验证
git clone https://github.com/ultralytics/yolov5
# 注释掉一些已有环境
pip install -r requirements.txt
python3 detect.py --weights yolov5s.pt --source data/images/bus.jpg --device 0

# 数据标注
# 标注工具1：labelimg
"""
python 3.9版本
C:\Users\Admin\anaconda3
安装labelimg;
conda activate base
# 数据标注准备
# yolo标签格式，保存为txt文件  五个值分别代表：类别，中心点横坐标，中心点纵坐标，检测框的宽，检测框的高。其中后四个数值都是经过归一化处理，为0-1的小数。
# PascalVOC标签格式，保存为xml文件
# CreateML标签格式，保存为json文件
pip install labelimg
"""
# 标注工具2：roboflow

# 训练
python train.py --data data/hayao.yaml --cfg models/hayao_yolov5s.yaml --weight pretrained/yolov5s.pt --epochs 50 --batch-size 16 --device 0,1
# 断点训练
python train.py --data data/hayao.yaml --cfg models/hayao_yolov5s.yaml --weight runs/train/exp/weights/last.pt --epochs 10 --batch-size 16 --device 0,1

# 测试
python detect.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train/exp2/weights/best.pt --source /root/dataset/public/object_detect/dataset_yolo_hayao/dataset/images/val/Image_20230310171432581.bmp --device 0
# Image_20230310171144605.bmp
python detect.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train/exp2/weights/best.pt --source /root/dataset/public/object_detect/dataset_yolo_hayao/dataset/images/val/Image_20230310171144605.bmp --device 0

python detect.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train/exp2/weights/best.pt --source /root/dataset/public/object_detect/dataset_yolo_hayao/dataset/images/val --device 0

# 转onnx
python export.py --data data/hayao.yaml --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train/exp2/weights/best.pt --include onnx --device 0 --opset 12

```



分类：

```python
python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224
```



## 二、源码模块分析











# What's New

