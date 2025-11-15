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
import time

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- SỬA ĐỔI QUAN TRỌNG ---
# Trỏ đến model mạnh nhất đã được fine-tune từ VisDrone backbone
SIAMESE_MODEL_PATH = '../pretrained_models/best_siamese_model_from_visdrone.pth'

PUBLIC_TEST_BASE_DIR = '../public_test/public_test/samples/'
OUTPUT_DIR = '../results/' # Đổi tên thư mục output để dễ phân biệt kết quả

# --- Siêu tham số có thể tinh chỉnh để có kết quả tốt hơn ---
IMG_SIZE = 128
THRESHOLD = 0.8         # Ngưỡng khoảng cách. GIẢM nếu phát hiện quá nhiều vật sai, TĂNG nếu bỏ sót vật thể.
NMS_THRESHOLD = 0.3     # Ngưỡng gộp box. GIẢM nếu còn nhiều box trùng, TĂNG nếu gộp nhầm box.
WINDOW_STEP = 32        # Bước nhảy. GIẢM để quét kỹ hơn (chậm hơn), TĂNG để quét nhanh hơn (có thể bỏ sót).
SCALES = [1.0]          # Thêm các scale khác như [0.75, 1.0, 1.25] để tìm vật thể ở các kích thước khác nhau.
BATCH_SIZE = 256        # Điều chỉnh tùy theo VRAM của GPU.

# --- 1. TẢI MODEL ---
print(f"--- Using device: {DEVICE} ---")
from siamese_network import SiameseNetwork 

print(f"Loading model from: {SIAMESE_MODEL_PATH}")
# Khởi tạo model với cấu trúc đúng, không cần tải lại backbone từ file khác
model = SiameseNetwork(backbone_path=None, pretrained_torch=False)
model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. LẤY DANH SÁCH VIDEO ---
# (Phần này đã tốt, giữ nguyên)
try:
    video_ids = [name for name in os.listdir(PUBLIC_TEST_BASE_DIR) if os.path.isdir(os.path.join(PUBLIC_TEST_BASE_DIR, name))]
except FileNotFoundError:
    print(f"ERROR: Base directory not found at '{PUBLIC_TEST_BASE_DIR}'. Please check the path.")
    exit()

print(f"Found {len(video_ids)} videos to process: {video_ids}")

# --- 3. VÒNG LẶP CHÍNH - XỬ LÝ TỪNG VIDEO ---
# (Toàn bộ logic xử lý video đã được tối ưu và đúng, giữ nguyên)
for video_id in video_ids:
    print(f"\n================ PROCESSING: {video_id} ================")
    
    test_data_dir = os.path.join(PUBLIC_TEST_BASE_DIR, video_id)
    reference_img_dir = os.path.join(test_data_dir, 'object_images')
    search_video_path = os.path.join(test_data_dir, 'drone_video.mp4')
    output_video_path = os.path.join(OUTPUT_DIR, f'{video_id}_result.mp4')

    if not (os.path.isdir(reference_img_dir) and os.path.isfile(search_video_path)):
        print(f"Warning: Necessary files not found for {video_id}. Skipping.")
        continue

    # --- 3b. Tạo vector tham chiếu trung bình ---
    print(f"Loading reference images...")
    ref_image_paths = glob.glob(os.path.join(reference_img_dir, '*.jpg')) + \
                      glob.glob(os.path.join(reference_img_dir, '*.png'))
    if not ref_image_paths:
        print(f"Warning: No reference images found. Skipping.")
        continue
    
    with torch.no_grad():
        ref_embeddings = [model.forward_one(transform(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)) for p in ref_image_paths]
        mean_ref_embedding = torch.mean(torch.cat(ref_embeddings, dim=0), dim=0, keepdim=True)

    # --- 3c. Xử lý video ---
    print(f"Processing video...")
    cap = cv2.VideoCapture(search_video_path)
    frame_width, frame_height, fps, frame_count = (int(cap.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    pbar = tqdm(total=frame_count, desc=f"Detecting in {video_id}")
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            detections = []
            windows_batch = []
            coords_batch = []

            for scale in SCALES:
                resized_h, resized_w = int(frame_height * scale), int(frame_width * scale)
                if resized_w < IMG_SIZE or resized_h < IMG_SIZE: continue
                
                resized_frame = cv2.resize(frame, (resized_w, resized_h))
                resized_frame_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

                for y in range(0, resized_h - IMG_SIZE, WINDOW_STEP):
                    for x in range(0, resized_w - IMG_SIZE, WINDOW_STEP):
                        window_pil = resized_frame_pil.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
                        windows_batch.append(transform(window_pil))
                        coords_batch.append((x, y, scale))

                        if len(windows_batch) >= BATCH_SIZE:
                            batch_tensor = torch.stack(windows_batch).to(DEVICE)
                            win_embeddings = model.forward_one(batch_tensor)
                            distances = F.pairwise_distance(mean_ref_embedding, win_embeddings)
                            
                            match_indices = torch.where(distances < THRESHOLD)[0]
                            for idx in match_indices.cpu().numpy():
                                cx, cy, cscale = coords_batch[idx]
                                score = 1.0 - (distances[idx].item() / THRESHOLD)
                                detections.append([int(cx/cscale), int(cy/cscale), int((cx+IMG_SIZE)/cscale), int((cy+IMG_SIZE)/cscale), score])
                            
                            windows_batch, coords_batch = [], []
            
            if windows_batch:
                batch_tensor = torch.stack(windows_batch).to(DEVICE)
                win_embeddings = model.forward_one(batch_tensor)
                distances = F.pairwise_distance(mean_ref_embedding, win_embeddings)
                match_indices = torch.where(distances < THRESHOLD)[0]
                for idx in match_indices.cpu().numpy():
                    cx, cy, cscale = coords_batch[idx]
                    score = 1.0 - (distances[idx].item() / THRESHOLD)
                    detections.append([int(cx/cscale), int(cy/cscale), int((cx+IMG_SIZE)/cscale), int((cy+IMG_SIZE)/cscale), score])

            if detections:
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

print("\n================ ALL VIDEOS PROCESSED ================")