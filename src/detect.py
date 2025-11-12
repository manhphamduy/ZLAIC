import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.ops import nms

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
SIAMESE_MODEL_PATH = '../pre-trained_models/best_siamese_model.pth'
REFERENCE_IMG_PATH = 'path/to/your/target/reference.jpg' # Cần một ảnh mẫu của mục tiêu
SEARCH_IMG_PATH = 'path/to/your/uav_frame.jpg'
THRESHOLD = 0.8  # Ngưỡng khoảng cách, cần tinh chỉnh!
NMS_THRESHOLD = 0.3 # Ngưỡng để loại bỏ box trùng lặp
WINDOW_STEP = 16
SCALES = [0.5, 0.75, 1.0, 1.25] # Quét ở nhiều kích thước

# --- 1. Tải mô hình và chuẩn bị ---
model = SiameseNetwork(backbone_path='dummy_path')
model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ref_img = Image.open(REFERENCE_IMG_PATH).convert("RGB")
ref_tensor = transform(ref_img).unsqueeze(0).to(DEVICE)

search_img_cv = cv2.imread(SEARCH_IMG_PATH)
(H, W) = search_img_cv.shape[:2]

# --- 2. Multi-scale Sliding Window ---
detections = []
with torch.no_grad():
    # Tính embedding cho ảnh tham chiếu một lần duy nhất
    ref_embedding = model.forward_one(ref_tensor)

    for scale in SCALES:
        resized_w, resized_h = int(W * scale), int(H * scale)
        if resized_w < IMG_SIZE or resized_h < IMG_SIZE: continue
        
        resized_img = cv2.resize(search_img_cv, (resized_w, resized_h))
        
        for y in range(0, resized_h - IMG_SIZE, WINDOW_STEP):
            for x in range(0, resized_w - IMG_SIZE, WINDOW_STEP):
                window_cv = resized_img[y:y + IMG_SIZE, x:x + IMG_SIZE]
                
                window_pil = Image.fromarray(cv2.cvtColor(window_cv, cv2.COLOR_BGR2RGB))
                window_tensor = transform(window_pil).unsqueeze(0).to(DEVICE)
                
                win_embedding = model.forward_one(window_tensor)
                distance = F.pairwise_distance(ref_embedding, win_embedding).item()
                
                if distance < THRESHOLD:
                    score = 1.0 - distance # Chuyển distance thành score (càng cao càng tốt)
                    box_x1 = int(x / scale)
                    box_y1 = int(y / scale)
                    box_x2 = int((x + IMG_SIZE) / scale)
                    box_y2 = int((y + IMG_SIZE) / scale)
                    detections.append([box_x1, box_y1, box_x2, box_y2, score])

# --- 3. Áp dụng Non-Maximum Suppression (NMS) ---
detections = np.array(detections)
if len(detections) > 0:
    boxes = torch.from_numpy(detections[:, :4]).float()
    scores = torch.from_numpy(detections[:, 4]).float()
    
    indices = nms(boxes, scores, NMS_THRESHOLD)
    final_boxes = boxes[indices].numpy().astype(int)
else:
    final_boxes = []

# --- 4. Vẽ kết quả cuối cùng ---
output_img = search_img_cv.copy()
for (startX, startY, endX, endY) in final_boxes:
    cv2.rectangle(output_img, (startX, startY), (endX, endY), (0, 255, 0), 3)

cv2.imshow("Final Detection", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()