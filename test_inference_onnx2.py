import onnxruntime as ort
import cv2
import numpy as np
import logging
import time
import os

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
        self.device = "cuda" if ort.get_device() == "GPU" else "cpu"

        # Load ONNX model
        self.weights = model_path
        self.session = ort.InferenceSession(self.weights, providers=['CUDAExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider'])
        
        self.imgsz = (640, 640)
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.max_det = 1000
        self.names = ['obj', '滑套', '弹桥', '螺丝', '螺母', '调节块']  # Replace with your class names
        
        logger.info(f"{self.class_name} init time is {time.time() - start_time}.")

    def postprocess(self, outputs, conf_thres=0.5, iou_thres=0.45):
        """
        Process the YOLO ONNX model outputs and apply non-max suppression.
        
        Args:
            outputs: Model output in the form of a list or numpy array.
            conf_thres: Confidence threshold to filter weak detections.
            iou_thres: IoU threshold for non-max suppression.
            
        Returns:
            selected_boxes: List of final bounding boxes after NMS.
            selected_scores: List of confidence scores for each box.
            selected_classes: List of class indices for each box.
        """
        # Unpack output (assuming outputs[0] contains the result)
        output = outputs[0][0]
        
        # Extract bounding box coordinates center_x, center_y, width, height
        boxes_pre = output[:, :4]
        boxes = np.zeros_like(boxes_pre)

        boxes[:, 0] = boxes_pre[:, 0] - boxes_pre[:, 2] / 2
        boxes[:, 1] = boxes_pre[:, 1] - boxes_pre[:, 3] / 2
        boxes[:, 2] = boxes_pre[:, 2]
        boxes[:, 3] = boxes_pre[:, 3]

        
        # Extract objectness score
        objectness = output[:, 4]
        
        # Extract class probabilities and calculate final scores (objectness * class probabilities)
        class_probs = output[:, 5:]
        class_probs = np.argmax(class_probs, axis=1)
        # Filter out boxes with confidence scores below the threshold
        valid_indices = np.where(objectness > conf_thres)[0]

        # print(f"valid_indices: {valid_indices}")
        selected_boxes = boxes[valid_indices]
        selected_scores = objectness[valid_indices]
        selected_classes = class_probs[valid_indices]
        
        # print(f"selected_boxes: {selected_boxes}")
        # print(f"selected_scores: {selected_scores}")

        # Convert selected_boxes to a list of lists for cv2.dnn.NMSBoxes
        boxes_for_nms = selected_boxes.tolist()
        scores_for_nms = selected_scores.tolist()

        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, score_threshold=conf_thres, nms_threshold=iou_thres)
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()
        else:
            indices = np.array([])  # If no boxes remain after NMS

        print(f"indices: {indices}")
        # Select final boxes, scores, and classes after NMS
        final_boxes = selected_boxes[indices]
        final_scores = selected_scores[indices]
        final_classes = selected_classes[indices]

        return final_boxes, final_scores, final_classes

    def inference(self, image, product_name, path_list=None, debug=False):
        start_time = time.time()
        labels = []
        points = []

        # Preprocess image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, (dw, dh) = letterbox(image, self.imgsz, stride=32, auto=True)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = np.expand_dims(image, axis=0).astype(np.float32)
        print(f"image: {image.shape}")
        # ONNX inference
        inputs = {self.session.get_inputs()[0].name: image}
        output = self.session.run(None, inputs)
        print(f"ort_outs: {output}")

        # Post-process output
        boxes, scores, classes = self.postprocess(output)
        print(f"boxes: {boxes}")
        print(f"scores: {scores}")
        print(f"classes: {classes}")

        # 遍历处理后的框，分数和类别
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            cls = int(classes[i])  # 类别索引
            label = self.names[cls]  # 获取类别名称

            x_s, y_s, w, h = box  # 获取检测框的坐标
            print(f"x_s: {x_s},  y_s: {y_s}, w: {w}, h: {h}, dw: {dw}, dh: {dh}, ratio: {ratio}")
            x_s = (x_s - dw) / ratio[0]
            y_s = (y_s - dh) / ratio[1]
            x_e = x_s + w / ratio[0]
            y_e = y_s + h / ratio[1]

            point = [int(x_s), int(y_s), int(x_e), int(y_e), float(score)]
            labels.append(label)
            points.append(point)
            print(f"Detected object: Class={label}, Box={point}, Score={score}")

        logger.info(f"{self.class_name} inference time is {time.time() - start_time}.")
        return labels, points


if __name__ == "__main__":
    # Initialize model
    model_path = "/root/project/research/Yolo/yolov5/runs/train/exp14/weights/best.onnx"  # Replace with your ONNX model path
    glass_inference = GlassInference(model_path)

    # Test image path
    image_path = "/root/dataset/glass_data/images/val/Image_20240725165715498.bmp"  # Replace with your test image path
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist, please check the path!")

    # Read test image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image {image_path}")
    
    # Run inference
    product_name = "test_product"
    labels, points = glass_inference.inference(image, product_name, path_list=None, debug=True)
    
    # Output inference results
    print(f"Detected labels: {labels}")
    print(f"Detected points: {points}")
    
    # Draw detection results
    for point in points:
        x_s, y_s, x_e, y_e, score = point
        cv2.rectangle(image, (x_s, y_s), (x_e, y_e), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x_s, y_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite("/root/dataset/glass_data/re_Image_20240725165715498.bmp", image)
