import cv2
import os
import random
import numpy as np

# 读取 YOLO 标注文件
def read_yolo_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            labels.append([float(x) for x in line.strip().split()])
    return labels

# 写入 YOLO 标注文件
def write_yolo_labels(file_path, labels):
    with open(file_path, 'w') as file:
        for label in labels:
            label[0] = int(label[0])
            file.write(' '.join([str(x) for x in label]) + '\n')

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=0.01):
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

# 调整对比度和亮度
def adjust_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 数据增强：旋转图像并更新标注
def augment_image_and_labels(image, labels, angle):
    h, w = image.shape[:2]

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # 更新标注
    new_labels = []
    for label in labels:
        class_id, cx, cy, bw, bh = label
        cx, cy = cx * w, cy * h  # 将归一化中心点转为像素坐标

        # 计算新的中心点坐标
        new_center = np.dot(M, np.array([cx, cy, 1]))
        new_cx, new_cy = new_center[0] / w, new_center[1] / h  # 归一化回原来范围
        new_labels.append([int(class_id), new_cx, new_cy, bw, bh])  # 宽高不变

    return rotated_image, new_labels

# 处理单个图像的增强
def process_image(image_path, label_path, output_dir, augmentations):
    image = cv2.imread(image_path)
    labels = read_yolo_labels(label_path)
    
    # 对每种增强方法进行处理并保存
    for augment_name, augment_func in augmentations.items():
        augmented_image, augmented_labels = augment_func(image, labels)

        # 保存增强后的图像和标注
        base_name = os.path.basename(image_path).split('.')[0]
        augmented_image_path = os.path.join(output_dir, f"{base_name}_{augment_name}.bmp")
        augmented_label_path = os.path.join(output_dir, f"{base_name}_{augment_name}.txt")

        cv2.imwrite(augmented_image_path, augmented_image)
        write_yolo_labels(augmented_label_path, augmented_labels)

# 增强方法字典
def get_augmentations():
    return {
        'gaussian_noise': lambda img, lbl: (add_gaussian_noise(img), lbl),
        'rotate_10': lambda img, lbl: augment_image_and_labels(img, lbl, angle=10),
        'rotate_-10': lambda img, lbl: augment_image_and_labels(img, lbl, angle=-10),
        'contrast_up': lambda img, lbl: (adjust_contrast(img, alpha=1.5, beta=0), lbl),
        'contrast_down': lambda img, lbl: (adjust_contrast(img, alpha=0.7, beta=0), lbl),
    }

# 批量处理图像和标注
def process_batch(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    augmentations = get_augmentations()

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.bmp'):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.bmp', '.txt'))
            process_image(image_path, label_path, output_dir, augmentations)

# 主函数
if __name__ == '__main__':
    image_dir = '/root/dataset/glass_data/images/train'  # 图像文件夹
    label_dir = '/root/dataset/glass_data/labels/train'  # YOLO 标注文件夹
    output_dir = '/root/dataset/glass_data/aug/'  # 增强后的输出文件夹

    # 开始批量处理
    process_batch(image_dir, label_dir, output_dir)
