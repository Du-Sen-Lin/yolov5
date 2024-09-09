import os
import cv2

# 定义裁剪区域和路径
image_dir = '/root/dataset/glass_data/images/train/'          # 图片目录
label_dir = '/root/dataset/glass_data/labels/train/'          # 标注文件目录
output_image_dir = '/root/dataset/glass_data/split_images/train/'  # 裁剪后图片存放目录
output_label_dir = '/root/dataset/glass_data/split_images/val/'  # 裁剪后标注文件存放目录

# 创建输出文件夹
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 裁剪区域
crop_x_min = 400  # 裁剪的起始x坐标
crop_y_min = 300  # 裁剪的起始y坐标
crop_width = 2200  # 裁剪区域宽度
crop_height = 2100  # 裁剪区域高度

def update_label(file_path, orig_width, orig_height, crop_x_min, crop_y_min, crop_width, crop_height):
    # 读取标注文件
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_labels = []
    for line in lines:
        # 获取每一行标注信息
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * orig_width
        y_center = float(parts[2]) * orig_height
        width = float(parts[3]) * orig_width
        height = float(parts[4]) * orig_height
        
        # 计算裁剪后物体位置
        new_x_center = x_center - crop_x_min
        new_y_center = y_center - crop_y_min
        
        # 如果中心点仍在裁剪区域内，更新标注
        if (0 <= new_x_center <= crop_width) and (0 <= new_y_center <= crop_height):
            new_x_center /= crop_width
            new_y_center /= crop_height
            new_width = width / crop_width
            new_height = height / crop_height
            
            # 添加新标注
            new_labels.append(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")
    
    return new_labels

# 遍历图片文件夹
for image_file in os.listdir(image_dir):
    if image_file.endswith('.bmp') or image_file.endswith('.png'):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.bmp', '.txt').replace('.png', '.txt'))
        
        # 读取图片
        img = cv2.imread(image_path)
        orig_height, orig_width = img.shape[:2]
        
        # 裁剪图片
        cropped_img = img[crop_y_min:crop_y_min+crop_height, crop_x_min:crop_x_min+crop_width]
        
        # 保存裁剪后的图片
        output_image_path = os.path.join(output_image_dir, image_file)
        cv2.imwrite(output_image_path, cropped_img)
        
        # 更新标注文件
        if os.path.exists(label_path):
            new_labels = update_label(label_path, orig_width, orig_height, crop_x_min, crop_y_min, crop_width, crop_height)
            
            # 保存新的标注文件
            output_label_path = os.path.join(output_label_dir, image_file.replace('.bmp', '.txt').replace('.png', '.txt'))
            with open(output_label_path, 'w') as f:
                f.writelines(new_labels)

print("裁剪与同步标注完成。")
