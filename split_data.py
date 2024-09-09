import os
import random
import shutil

def split_data(train_img_dir, val_img_dir, train_txt_dir, val_txt_dir, image_ext=".bmp", label_ext=".txt", val_ratio=0.2):
    # 获取所有图片和标注文件的路径
    images = [f for f in os.listdir(train_img_dir) if f.endswith(image_ext)]
    
    # 打乱图片列表
    random.shuffle(images)
    
    # 按比例划分训练集和验证集
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    train_images = images[val_count:]
    
    # 将验证集文件移动到验证集目录
    for image in val_images:
        image_path = os.path.join(train_img_dir, image)
        label_path = os.path.join(train_txt_dir, image.replace(image_ext, label_ext))
        
        shutil.move(image_path, os.path.join(val_img_dir, image))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_txt_dir, image.replace(image_ext, label_ext)))
    
    print(f"验证集数量: {val_count}")
    print(f"训练集数量: {len(train_images)}")

# 使用示例
train_img_dir = "/root/dataset/glass_data/images/train"  # 训练集目录
val_img_dir = "/root/dataset/glass_data/images/val"      # 验证集目录
train_txt_dir = "/root/dataset/glass_data/labels/train"  # 训练集目录
val_txt_dir = "/root/dataset/glass_data/labels/val"      # 验证集目录
val_ratio = 0.2               # 验证集比例 (20%)

split_data(train_img_dir, val_img_dir, train_txt_dir, val_txt_dir, val_ratio=val_ratio)
