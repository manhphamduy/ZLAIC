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
import json

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- THAY ĐỔI QUAN TRỌNG NHẤT ---
# Đường dẫn đến model TỐT NHẤT đã được fine-tune từ VisDrone backbone
SIAMESE_MODEL_PATH = '../pre-trained_models/best_siamese_model_from_visdrone.pth'

# Đường dẫn đến thư mục chứa tất cả video test
PUBLIC_TEST_BASE_DIR = '../public_test/public_test/samples/'
# Tên file JSON output để nộp bài
OUTPUT_JSON_PATH = '../submission.json'

# Siêu tham số cho phát hiện (CẦN TINH CHỈNH KỸ LƯỠNG)
IMG_SIZE = 128
THRESHOLD = 0.8        # Ngưỡng khoảng cách, có thể cần giảm/tăng tùy theo kết quả
NMS_THRESHOLD = 0.3    # Ngưỡng để loại bỏ box trùng lặp
WINDOW_STEP = 32       # Bước nhảy của cửa sổ, tăng để nhanh hơn, giảm để kỹ hơn
BATCH_SIZE = 256       # Số cửa sổ xử lý cùng lúc trên GPU

# --- 1. TẢI MODEL (Chỉ tải một lần duy nhất) ---
print(f"--- Using device: {DEVICE} ---")
# Import class SiameseNetwork từ file siamese_network.py
from siamese_network import SiameseNetwork 

print(f"Loading model from: {SIAMESE_MODEL_PATH}")
# Khởi tạo model với cấu trúc đúng (không cần tải pre-trained weights từ file khác nữa)
model = SiameseNetwork(backbone_path=None, pretrained_torch=False)
model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. LẤY DANH SÁCH VIDEO VÀ CHUẨN BỊ ---
try:
    video_ids = [name for name in os.listdir(PUBLIC_TEST_BASE_DIR) if os.path.isdir(os.path.join(PUBLIC_TEST_BASE_DIR, name))]
except FileNotFoundError:
    print(f"ERROR: Base directory not found at '{PUBLIC_TEST_BASE_DIR}'. Please check the path.")
    exit()

print(f"Found {len(video_ids)} videos to process: {video_ids}")
all_results = [] # Danh sách chứa kết quả cuối cùng của tất cả video

# --- 3. VÒNG LẶP CHÍNH - XỬ LÝ TỪNG VIDEO ---
# (Toàn bộ phần logic xử lý video bên dưới được giữ nguyên vì đã tối ưu)
for video_id in video_ids:
    print(f"\n================ PROCESSING: {video_id} ================")
    
    test_data_dir = os.path.join(PUBLIC_TEST_BASE_DIR, video_id)
    reference_img_dir = os.path.join(test_data_dir, 'object_images')
    search_video_path = os.path.join(test_data_dir, 'drone_video.mp4')
    
    if not (os.path.isdir(reference_img_dir) and os.path.isfile(search_video_path)):
        print(f"Warning: Necessary files not found for {video_id}. Skipping.")
        all_results.append({"video_id": video_id, "annotations": [{"bboxes": []}]})
        continue

    # --- 3b. Tạo vector tham chiếu trung bình ---
    ref_image_paths = glob.glob(os.path.join(reference_img_dir, '*.jpg')) + \
                      glob.glob(os.path.join(reference_img_dir, '*.png'))
    if not ref_image_paths:
        print(f"Warning: No reference images found. Skipping video {video_id}.")
        all_results.append({"video_id": video_id, "annotations": [{"bboxes": []}]})
        continue
    
    with torch.no_grad():
        ref_embeddings = [model.forward_one(transform(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)) for p in ref_image_paths]
        mean_ref_embedding = torch.mean(torch.cat(ref_embeddings, dim=0), dim=0, keepdim=True)

    # --- 3c. Xử lý video và lưu kết quả bboxes ---
    cap = cv2.VideoCapture(search_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_annotations = {"bboxes": []}
    current_frame_idx = 0
    
    pbar = tqdm(total=frame_count, desc=f"Inferencing on {video_id}")
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_height, frame_width = frame.shape[:2]
            detections = []
            
            # --- Tối ưu hóa bằng Batch Processing ---
            windows_batch = []
            coords_batch = []
            resized_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            for y in range(0, frame_height - IMG_SIZE, WINDOW_STEP):
                for x in range(0, frame_width - IMG_SIZE, WINDOW_STEP):
                    window_pil = resized_frame_pil.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
                    windows_batch.append(transform(window_pil))
                    coords_batch.append((x, y))

                    if len(windows_batch) >= BATCH_SIZE:
                        batch_tensor = torch.stack(windows_batch).to(DEVICE)
                        win_embeddings = model.forward_one(batch_tensor)
                        distances = F.pairwise_distance(mean_ref_embedding, win_embeddings)
                        
                        match_indices = torch.where(distances < THRESHOLD)[0]
                        for idx in match_indices.cpu().numpy():
                            cx, cy = coords_batch[idx]
                            score = 1.0 - (distances[idx].item() / THRESHOLD)
                            detections.append([cx, cy, cx + IMG_SIZE, cy + IMG_SIZE, score])
                        windows_batch, coords_batch = [], []
            
            if windows_batch: # Xử lý phần còn lại
                batch_tensor = torch.stack(windows_batch).to(DEVICE)
                win_embeddings = model.forward_one(batch_tensor)
                distances = F.pairwise_distance(mean_ref_embedding, win_embeddings)
                match_indices = torch.where(distances < THRESHOLD)[0]
                for idx in match_indices.cpu().numpy():
                    cx, cy = coords_batch[idx]
                    score = 1.0 - (distances[idx].item() / THRESHOLD)
                    detections.append([cx, cy, cx + IMG_SIZE, cy + IMG_SIZE, score])

            # --- Áp dụng NMS và LƯU KẾT QUẢ ---
            if detections:
                detections_np = np.array(detections)
                boxes_tensor = torch.from_numpy(detections_np[:, :4]).float()
                scores_tensor = torch.from_numpy(detections_np[:, 4]).float()
                indices = nms(boxes_tensor, scores_tensor, NMS_THRESHOLD)
                final_boxes = boxes_tensor[indices].numpy().astype(int)
                
                for (x1, y1, x2, y2) in final_boxes:
                    bbox_dict = {
                        "frame": current_frame_idx, "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    }
                    video_annotations["bboxes"].append(bbox_dict)

            current_frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    
    video_result_object = {
        "video_id": video_id, "annotations": [video_annotations]
    }
    all_results.append(video_result_object)
    print(f"--- Finished {video_id}. Found {len(video_annotations['bboxes'])} bounding boxes. ---")


# --- 4. LƯU KẾT QUẢ CUỐI CÙNG RA FILE JSON ---
print("\n================ SAVING FINAL SUBMISSION FILE ================")
with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Successfully created submission file at: {OUTPUT_JSON_PATH}")