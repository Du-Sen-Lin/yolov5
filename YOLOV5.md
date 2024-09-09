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
python val.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train/exp2/weights/best.pt --data data/hayao.yaml --batch-size 4
"""
        Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 82/82 [00:18<00:00,  4.37it/s]
           all        327        210      0.859      0.887      0.943      0.667
   abnormal_ob        327         45      0.844          1      0.992      0.751
       plastic        327         36      0.957          1      0.995      0.777
        insect        327         25      0.906          1      0.995      0.772
     black_dot        327         23      0.782      0.435      0.737      0.274
       stomium        327         81      0.805          1      0.994      0.763
Speed: 0.2ms pre-process, 3.2ms inference, 0.6ms NMS per image at shape (4, 3, 640, 640)
"""

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

python classify/val.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train-cls/exp3/weights/best.pt --data ../datasets/imagenette160 --img 224
"""
   Class      Images    top1_acc    top5_acc
     all        3925        0.96       0.997
n01440764         387       0.969           1
n02102040         395       0.975       0.997
n02979186         357       0.966       0.994
n03000684         386        0.94       0.995
n03028079         409       0.978           1
n03394916         394       0.949       0.997
n03417042         389       0.967       0.997
n03425413         419       0.921       0.998
n03445777         399        0.97       0.997
n03888257         390       0.969       0.997
Speed: 0.2ms pre-process, 0.2ms inference, 0.0ms post-process per image at shape (1, 3, 224, 224)
"""

python classify/predict.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train-cls/exp3/weights/best.pt --source /root/project/bp_algo/common/YOLO/datasets/imagenette160/val/n01440764

python export.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train-cls/exp3/weights/best.pt --include onnx --device 0 --opset 12 --imgsz 224

python classify/val.py --weights /root/project/bp_algo/common/YOLO/yolov5/runs/train-cls/exp3/weights/best.onnx --data /root/project/bp_algo/common/YOLO/datasets/imagenette160 --batch-size 4 --imgsz 224

"""
   Class      Images    top1_acc    top5_acc
     all        3925        0.96       0.997
n01440764         387       0.969           1
n02102040         395       0.975       0.997
n02979186         357       0.966       0.994
n03000684         386        0.94       0.995
n03028079         409       0.978           1
n03394916         394       0.949       0.997
n03417042         389       0.967       0.997
n03425413         419       0.921       0.998
n03445777         399        0.97       0.997
n03888257         390       0.969       0.997
Speed: 0.1ms pre-process, 2.0ms inference, 0.1ms post-process per image at shape (1, 3, 224, 224)
"""
```



## 二、源码模块分析

notebook:

```python
# 00_module_knowledge_points.ipynb

# python train.py --data data/hayao.yaml --cfg models/hayao_yolov5s.yaml --weight runs/train/exp2/weights/last.pt --epochs 1 --batch-size 16 --device 0,1
```



# What's New

