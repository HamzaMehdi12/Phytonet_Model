import cv2
import numpy as np
import torch
import onnxruntime as ort
from torchvision.ops import nms

class TomatoDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = 320
        self.class_names = ['tomato', 'stem', 'stem-tomato']  # Updated to 3 classes
        
    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img
        
    def decode_predictions(self, outputs, conf_thresh=0.35, iou_thresh=0.45):
        # Process output tensor with 3 anchors and 3 classes
        pred = outputs[0][0]
        A = 3  # 3 anchors now
        C, H, W = pred.shape
        num_classes = 3  # 3 classes
        
        # Reshape and process
        pred = pred.reshape(A, 5 + num_classes, H, W).transpose(0, 2, 3, 1)
        obj_conf = 1 / (1 + np.exp(-pred[..., 4]))
        mask = obj_conf > conf_thresh
        
        if not np.any(mask):
            return [], [], []
            
        anchors, ys, xs = np.where(mask)
        p = pred[anchors, ys, xs]
        
        # Decode boxes
        grid_size_x = self.img_size / W
        grid_size_y = self.img_size / H
        px = 1 / (1 + np.exp(-p[:, 0]))
        py = 1 / (1 + np.exp(-p[:, 1]))
        pconf = obj_conf[anchors, ys, xs]
        pcls = np.argmax(p[:, 5:5+num_classes], axis=1)  # Only first 3 classes
        
        cx = (xs.astype(np.float32) + px) * grid_size_x
        cy = (ys.astype(np.float32) + py) * grid_size_y
        bw = np.exp(p[:, 2]) * grid_size_x
        bh = np.exp(p[:, 3]) * grid_size_y
        
        x1 = cx - bw/2
        y1 = cy - bh/2
        x2 = cx + bw/2
        y2 = cy + bh/2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = pconf
        
        # Apply NMS
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes)
            scores_tensor = torch.tensor(scores)
            keep = nms(boxes_tensor, scores_tensor, iou_thresh)
            boxes = boxes[keep.numpy()]
            scores = scores[keep.numpy()]
            classes = pcls[keep.numpy()]
            
        return boxes, scores, classes
    
    def detect(self, img):
        input_data = self.preprocess(img)[np.newaxis, ...]
        outputs = self.session.run(None, {self.input_name: input_data})
        return self.decode_predictions(outputs)

    def draw_detections(self, img, boxes, scores, classes):
        h, w = img.shape[:2]
        scale_x, scale_y = w / self.img_size, h / self.img_size
        
        for box, score, cls in zip(boxes, scores, classes):
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            color = (0, 255, 0)  # Green for tomato
            if cls == 1:  # Stem
                color = (0, 0, 255)  # Red
            elif cls == 2:  # Stem-tomato
                color = (255, 0, 0)  # Blue
                
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{self.class_names[cls]}: {score:.2f}"
            cv2.putText(img, label, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

if __name__ == '__main__':
    detector = TomatoDetector("phytonet.onnx")
    cap = cv2.VideoCapture(0)  # Use webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        boxes, scores, classes = detector.detect(frame)
        result = detector.draw_detections(frame, boxes, scores, classes)
        
        cv2.imshow('Botanical Detector', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()