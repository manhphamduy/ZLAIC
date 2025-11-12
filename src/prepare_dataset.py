import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- Cấu hình ---
RAW_DATA_DIR = '../data/dataset/'
PROCESSED_DATA_DIR = '../data/processed_rescue_data/'
ANNOTATIONS_FILE = os.path.join(RAW_DATA_DIR, 'annotations', 'train.json') # Thay 'train.json' bằng tên file của bạn
SAMPLES_DIR = os.path.join(RAW_DATA_DIR, 'samples')
VIDEO_EXTENSION = '.mp4'  # Giả định video có đuôi .mp4

# Các tham số để trích xuất background
MIN_BG_SIZE = 64
NUM_BG_PER_OBJECT = 3 # Lấy 3 ảnh nền cho mỗi đối tượng trong frame

# --- Tạo thư mục output ---
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
bg_dir = os.path.join(PROCESSED_DATA_DIR, 'background')
if not os.path.exists(bg_dir):
    os.makedirs(bg_dir)

print("--- Starting Video Data Preparation ---")
object_counts = defaultdict(int)
bg_count = 0

# 1. Đọc và tải file annotation
with open(ANNOTATIONS_FILE, 'r') as f:
    all_videos_data = json.load(f)

# 2. Lặp qua từng video trong file JSON
for video_data in tqdm(all_videos_data, desc="Processing Videos"):
    video_id = video_data['video_id']
    video_path = os.path.join(SAMPLES_DIR, video_id + VIDEO_EXTENSION)

    if not os.path.exists(video_path):
        print(f"\nWarning: Video file not found for {video_id}, skipping.")
        continue

    # 3. Restructure annotations: Nhóm các bounding box theo frame
    frames_data = defaultdict(list)
    # Suy ra tên class từ video_id, vd: "Backpack_0" -> "Backpack"
    class_name = "_".join(video_id.split('_')[:-1])
    if not class_name: # Nếu video_id không có dấu '_', dùng luôn video_id
        class_name = video_id
        
    for tracked_object in video_data['annotations']:
        for bbox in tracked_object['bboxes']:
            frame_idx = bbox['frame']
            box_coords = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
            frames_data[frame_idx].append({
                'box': box_coords,
                'class': class_name
            })

    # 4. Mở video và xử lý từng frame
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    while cap.isOpened():
        ret, frame_img = cap.read()
        if not ret:
            break

        # Nếu frame hiện tại có annotation
        if current_frame in frames_data:
            objects_in_frame = frames_data[current_frame]
            h, w, _ = frame_img.shape
            object_mask = np.zeros((h, w), dtype=np.uint8)

            # 4a. Cắt các đối tượng (target)
            for obj in objects_in_frame:
                class_name = obj['class']
                x1, y1, x2, y2 = obj['box']
                
                # Đảm bảo tọa độ nằm trong ảnh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Cắt ảnh
                cropped_obj = frame_img[y1:y2, x1:x2]

                if cropped_obj.size > 0:
                    class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                    
                    save_path = os.path.join(class_dir, f"{video_id}_f{current_frame}_{object_counts[class_name]}.jpg")
                    cv2.imwrite(save_path, cropped_obj)
                    object_counts[class_name] += 1
                
                # Cập nhật mask để không lấy background ở vùng này
                cv2.rectangle(object_mask, (x1, y1), (x2, y2), 255, -1)

            # 4b. Cắt các vùng nền (background)
            for _ in range(len(objects_in_frame) * NUM_BG_PER_OBJECT):
                bg_h = np.random.randint(MIN_BG_SIZE, h // 4)
                bg_w = np.random.randint(MIN_BG_SIZE, w // 4)
                
                # Thử tìm một vị trí hợp lệ trong 10 lần
                for _ in range(10):
                    x = np.random.randint(0, w - bg_w)
                    y = np.random.randint(0, h - bg_h)
                    
                    if np.sum(object_mask[y:y+bg_h, x:x+bg_w]) == 0:
                        cropped_bg = frame_img[y:y+bg_h, x:x+bg_w]
                        if cropped_bg.size > 0:
                            save_path = os.path.join(bg_dir, f"{video_id}_f{current_frame}_{bg_count}.jpg")
                            cv2.imwrite(save_path, cropped_bg)
                            bg_count += 1
                        break # Tìm được rồi thì thoát

        current_frame += 1
    
    cap.release()

print("\n--- Data Preparation Finished ---")
print(f"Total background images extracted: {bg_count}")
for class_name, count in object_counts.items():
    print(f"Total '{class_name}' images extracted: {count}")