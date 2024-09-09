import onnxruntime as ort
import numpy as np
import cv2
import time

# Load ONNX model
onnx_model_path = "/root/project/research/Yolo/yolov5/runs/train/exp14/weights/best.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Helper function to preprocess the image
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def preprocess(image_path, img_size=(640, 640)):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img, _, _ = letterbox(img, new_shape=img_size)  # Resize image and pad it
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)  # Convert to float32
    img /= 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess(outputs, conf_thres=0.25, iou_thres=0.45):
    """
    Process ONNX outputs and apply non-max suppression.
    """
    boxes = outputs[0][:, :4]  # x1, y1, x2, y2
    scores = outputs[0][:, 4] * outputs[0][:, 5:]  # confidence * class probability
    max_scores = np.max(scores, axis=1)  # get max score for each box
    valid_indices = np.where(max_scores > conf_thres)[0]  # filter by conf threshold

    selected_boxes = boxes[valid_indices]
    selected_scores = max_scores[valid_indices]
    return selected_boxes, selected_scores

def run_inference(image_path):
    # Preprocess the image
    img = preprocess(image_path)

    # Run inference
    start_time = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")

    # Post-process the output
    boxes, scores = postprocess(ort_outs)
    return boxes, scores

if __name__ == "__main__":
    image_path = "/root/dataset/glass_data/images/val/Image_20240725165715498.bmp"
    boxes, scores = run_inference(image_path)

    # Print out the results
    print("Detected boxes:", boxes)
    print("Confidence scores:", scores)
