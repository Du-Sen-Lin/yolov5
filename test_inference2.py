from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
import logging
import time
import os
import numpy as np
import torch


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(levelname)s]: %(message)s')
logger = logging.getLogger()
os.environ["CUDA_VISBLE_DEVICES"] = "0"


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class GlassInference:
    def __init__(self, model_path):
        start_time = time.time()
        self.class_name = self.__class__.__name__
        logger.info(f"{self.class_name} model_path: {model_path}")
        self.device = 0

        # self.weights = "/root/project/research/Yolo/yolov5/runs/train/exp14/weights/best.pt"
        self.weights = "/root/project/research/Yolo/yolov5/runs/train/exp14/weights/best.onnx"
        self.data = "/root/project/research/Yolo/yolov5/data/glass.yaml"
        self.imgsz = (640, 640)
        self.augment = False
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000

        # load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        # self.model = DetectMultiBackend(
        #     self.weights, device=self.device, dnn=False, data=self.data, fp16=True)        
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        logger.info(
            f"self.stride: {self.stride} \n self.names: {self.names} \n self.pt: {self.pt}")
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        logger.info(
            f"{self.class_name} init time is {time.time() - start_time}.")


    def inference(self, image, product_name, path_list=None, debug=False):
        start_time = time.time()
        labels = []
        points = []

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, (dw, dh)= letterbox(image, self.imgsz, stride=self.stride, auto=True)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).to(self.model.device)
        image = image.half() if self.model.fp16 else image.float()
        image /= 255
        if len(image.shape) == 3:
            image = image[None]
        y_pred = self.model(image)
        y_pred_f = non_max_suppression(y_pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        for t in range(len(y_pred_f[0])):
            re = y_pred_f[0][t]
            print(f"re: {re}")
            x_s, y_s, x_e, y_e, score, cls = re[:]
            print(f"x_s: {x_s}, y_s: {y_s}, x_e: {x_e}, y_e: {y_e}, dw: {dw}, dh: {dh}, ratio: {ratio}")
            x_s = (x_s - dw) / ratio[0]
            y_s = (y_s - dh) / ratio[1]
            x_e = (x_e - dw) / ratio[0]
            y_e = (y_e - dh) / ratio[1]

            print(f"self names: {self.names}")

            label = self.names[int(cls)]
            point = [int(x_s), int(y_s), 
                        int(x_e), int(y_e), float(score)]
            labels.append(label)
            points.append(point)

        logger.info(f"{self.class_name} inference time is {time.time() - start_time}.")

        return labels, points


if __name__ == "__main__":
    # 初始化模型
    model_path = "/root/project/research/Yolo/yolov5/runs/train/exp14/weights/best.pt"  # 替换为你的模型路径
    glass_inference = GlassInference(model_path)

    # 测试图像路径
    image_path = "/root/dataset/glass_data/images/val/Image_20240725165715498.bmp"  # 替换为你的测试图像路径
    if not os.path.exists(image_path):
        print(f"图像 {image_path} 不存在，请检查路径！")
    
    # 读取测试图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
    
    # 运行推理
    product_name = "test_product"
    labels, points = glass_inference.inference(image, product_name, path_list=None, debug=True)
    
    # 输出推理结果
    print(f"检测到的标签: {labels}")
    print(f"检测到的坐标点: {points}")
    
    # 绘制检测结果
    for point in points:
        x_s, y_s, x_e, y_e, score = point
        cv2.rectangle(image, (x_s, y_s), (x_e, y_e), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x_s, y_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite("/root/dataset/glass_data/re_Image_20240725165715498.bmp", image)