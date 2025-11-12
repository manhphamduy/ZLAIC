import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.ops import nms
import os
import glob
from tqdm import tqdm

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Đường dẫn đến model đã train (sử dụng MobileNetV3)
SIAMESE_MODEL_PATH = '../pretrained_models/best_siamese_model_mobilenet_v3.pth'

# --- ĐƯỜNG DẪN ĐẾN THƯ MỤC CHA CHỨA TẤT CẢ VIDEO TEST ---
PUBLIC_TEST_BASE_DIR = '../public_test/public_test/samples/'

# Đường dẫn lưu tất cả video kết quả
OUTPUT_DIR = '../results/'

# Siêu tham số cho phát hiện
IMG_SIZE = 128
THRESHOLD = 0.8        # Ngưỡng khoảng cách, CẦN TINH CHỈNH!
NMS_THRESHOLD = 0.3    # Ngưỡng để loại bỏ box trùng lặp
WINDOW_STEP = 16       # Bước nhảy của cửa sổ trượt
SCALES = [1.0]         # Bắt đầu với 1 scale để chạy nhanh hơn

# --- 1. TẢI MODEL (Chỉ tải một lần duy nhất) ---
print(f"--- Using device: {DEVICE} ---")
# Import class SiameseNetwork từ file siamese_network.py
from siamese_network import SiameseNetwork 

print(f"Loading model from: {SIAMESE_MODEL_PATH}")
model = SiameseNetwork(pretrained=False) # pretrained=False vì ta sẽ load state_dict
model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval() # Chuyển sang chế độ inference

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. LẤY DANH SÁCH TẤT CẢ VIDEO CẦN XỬ LÝ ---
try:
    video_ids = [name for name in os.listdir(PUBLIC_TEST_BASE_DIR) if os.path.isdir(os.path.join(PUBLIC_TEST_BASE_DIR, name))]
except FileNotFoundError:
    print(f"ERROR: Base directory not found at '{PUBLIC_TEST_BASE_DIR}'. Please check the path.")
    exit()

print(f"Found {len(video_ids)} videos to process: {video_ids}")


# --- 3. VÒNG LẶP CHÍNH - XỬ LÝ TỪNG VIDEO ---
for video_id in video_ids:
    print(f"\n================ PROCESSING: {video_id} ================")
    
    # --- 3a. Xác định đường dẫn cho video hiện tại ---
    test_data_dir = os.path.join(PUBLIC_TEST_BASE_DIR, video_id)
    reference_img_dir = os.path.join(test_data_dir, 'object_images')
    search_video_path = os.path.join(test_data_dir, 'drone_video.mp4')
    output_video_path = os.path.join(OUTPUT_DIR, f'{video_id}_result.mp4')
    
    # Kiểm tra xem các file/thư mục cần thiết có tồn tại không
    if not os.path.isdir(reference_img_dir):
        print(f"Warning: 'object_images' directory not found for {video_id}. Skipping.")
        continue
    if not os.path.isfile(search_video_path):
        print(f"Warning: 'drone_video.mp4' not found for {video_id}. Skipping.")
        continue

    # --- 3b. Tạo vector đặc trưng tham chiếu TRUNG BÌNH cho video hiện tại ---
    print(f"Loading reference images from: {reference_img_dir}")
    ref_image_paths = glob.glob(os.path.join(reference_img_dir, '*.jpg')) + \
                      glob.glob(os.path.join(reference_img_dir, '*.png'))

    if not ref_image_paths:
        print(f"Warning: No reference images found in {reference_img_dir}. Skipping.")
        continue

    ref_embeddings = []
    with torch.no_grad():
        for img_path in ref_image_paths:
            ref_img = Image.open(img_path).convert("RGB")
            ref_tensor = transform(ref_img).unsqueeze(0).to(DEVICE)
            embedding = model.forward_one(ref_tensor)
            ref_embeddings.append(embedding)

    mean_ref_embedding = torch.mean(torch.cat(ref_embeddings, dim=0), dim=0, keepdim=True)

    # --- 3c. Xử lý video và phát hiện ---
    print(f"Processing video: {search_video_path}")
    cap = cv2.VideoCapture(search_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pbar = tqdm(total=frame_count, desc=f"Detecting in {video_id}")
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = []
            for scale in SCALES:
                # ... (Phần sliding window và NMS giữ nguyên) ...
                resized_h, resized_w = int(frame_height * scale), int(frame_width * scale)
                if resized_w < IMG_SIZE or resized_h < IMG_SIZE: continue
                resized_frame = cv2.resize(frame, (resized_w, resized_h))
                for y in range(0, resized_h - IMG_SIZE, WINDOW_STEP):
                    for x in range(0, resized_w - IMG_SIZE, WINDOW_STEP):
                        window_cv = resized_frame[y:y + IMG_SIZE, x:x + IMG_SIZE]
                        window_pil = Image.fromarray(cv2.cvtColor(window_cv, cv2.COLOR_BGR2RGB))
                        window_tensor = transform(window_pil).unsqueeze(0).to(DEVICE)
                        win_embedding = model.forward_one(window_tensor)
                        distance = F.pairwise_distance(mean_ref_embedding, win_embedding).item()
                        if distance < THRESHOLD:
                            score = 1.0 - (distance / THRESHOLD)
                            box_x1 = int(x / scale)
                            box_y1 = int(y / scale)
                            box_x2 = int((x + IMG_SIZE) / scale)
                            box_y2 = int((y + IMG_SIZE) / scale)
                            detections.append([box_x1, box_y1, box_x2, box_y2, score])

            final_boxes = []
            if len(detections) > 0:
                detections_np = np.array(detections)
                boxes_tensor = torch.from_numpy(detections_np[:, :4]).float()
                scores_tensor = torch.from_numpy(detections_np[:, 4]).float()
                indices = nms(boxes_tensor, scores_tensor, NMS_THRESHOLD)
                final_boxes = boxes_tensor[indices].numpy().astype(int)

            for (startX, startY, endX, endY) in final_boxes:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            writer.write(frame)
            pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"--- Finished {video_id}. Result video saved to: {output_video_path} ---")

# --- 4. HOÀN TẤT ---
print("\n================ ALL VIDEOS PROCESSED ================")