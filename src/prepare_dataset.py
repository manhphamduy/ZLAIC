import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import shutil # Thư viện để copy file
import glob   # Thư viện để tìm kiếm file

# --- Cấu hình ---
RAW_DATA_DIR = '../data/dataset/'
PROCESSED_DATA_DIR = '../data/processed_rescue_data/'
ANNOTATIONS_FILE = os.path.join(RAW_DATA_DIR, 'annotations', 'annotations.json') # Thay 'train.json' bằng tên file của bạn
SAMPLES_DIR = os.path.join(RAW_DATA_DIR, 'samples')

# Các tham số để trích xuất background
MIN_BG_SIZE = 64
NUM_BG_PER_OBJECT = 3

# --- Tạo thư mục output ---
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
bg_dir = os.path.join(PROCESSED_DATA_DIR, 'background')
if not os.path.exists(bg_dir):
    os.makedirs(bg_dir)

print("--- Starting Hybrid Data Preparation (Images + Video) ---")
object_counts = defaultdict(int)
bg_count = 0

# 1. Đọc và tải file annotation
with open(ANNOTATIONS_FILE, 'r') as f:
    all_videos_data = json.load(f)

# 2. Lặp qua từng video trong file JSON
for video_data in tqdm(all_videos_data, desc="Processing Videos and Images"):
    video_id = video_data['video_id'] # vd: "Backpack_0"
    video_folder_path = os.path.join(SAMPLES_DIR, video_id)

    if not os.path.isdir(video_folder_path):
        print(f"\nWarning: Directory not found for {video_id}, skipping.")
        continue

    # Suy ra tên class từ video_id, vd: "Backpack_0" -> "Backpack"
    class_name = "_".join(video_id.split('_')[:-1])
    if not class_name:
        class_name = video_id
        
    class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # 3. TẬN DỤNG "KHO BÁU" object_images
    object_images_dir = os.path.join(video_folder_path, 'object_images')
    if os.path.isdir(object_images_dir):
        for img_file in os.listdir(object_images_dir):
            src_path = os.path.join(object_images_dir, img_file)
            dst_path = os.path.join(class_dir, f"{video_id}_objimg_{object_counts[class_name]}.jpg")
            shutil.copy(src_path, dst_path)
            object_counts[class_name] += 1

    # 4. TÌM VIDEO VÀ XỬ LÝ
    # Tìm file video (.mp4, .avi, ...) trong thư mục video_id
    # Cách này linh hoạt hơn là giả định tên file là 'drone_video.mp4'
    video_files = glob.glob(os.path.join(video_folder_path, '*.mp4')) + \
                  glob.glob(os.path.join(video_folder_path, '*.avi')) + \
                  glob.glob(os.path.join(video_folder_path, '*.mov'))

    if not video_files:
        print(f"\nWarning: No video file found inside {video_folder_path}, skipping video processing.")
        continue
    
    video_path = video_files[0] # Lấy video đầu tiên tìm thấy

    # 5. Restructure annotations cho video
    frames_data = defaultdict(list)
    for tracked_object in video_data['annotations']:
        for bbox in tracked_object['bboxes']:
            frame_idx = bbox['frame']
            box_coords = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
            frames_data[frame_idx].append({'box': box_coords})

    # 6. Mở video và xử lý từng frame
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    while cap.isOpened():
        ret, frame_img = cap.read()
        if not ret:
            break

        if current_frame in frames_data:
            objects_in_frame = frames_data[current_frame]
            h, w, _ = frame_img.shape
            object_mask = np.zeros((h, w), dtype=np.uint8)

            # 6a. Cắt các đối tượng từ video
            for obj in objects_in_frame:
                x1, y1, x2, y2 = obj['box']
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                cropped_obj = frame_img[y1:y2, x1:x2]

                if cropped_obj.size > 0:
                    save_path = os.path.join(class_dir, f"{video_id}_f{current_frame}_{object_counts[class_name]}.jpg")
                    cv2.imwrite(save_path, cropped_obj)
                    object_counts[class_name] += 1
                
                cv2.rectangle(object_mask, (x1, y1), (x2, y2), 255, -1)

            # 6b. Cắt các vùng nền từ video
            for _ in range(len(objects_in_frame) * NUM_BG_PER_OBJECT):
                bg_h = np.random.randint(MIN_BG_SIZE, h // 4)
                bg_w = np.random.randint(MIN_BG_SIZE, w // 4)
                for _ in range(10):
                    x = np.random.randint(0, w - bg_w)
                    y = np.random.randint(0, h - bg_h)
                    if np.sum(object_mask[y:y+bg_h, x:x+bg_w]) == 0:
                        cropped_bg = frame_img[y:y+bg_h, x:x+bg_w]
                        if cropped_bg.size > 0:
                            save_path = os.path.join(bg_dir, f"{video_id}_f{current_frame}_{bg_count}.jpg")
                            cv2.imwrite(save_path, cropped_bg)
                            bg_count += 1
                        break

        current_frame += 1
    
    cap.release()

print("\n--- Data Preparation Finished ---")
print(f"Total background images extracted: {bg_count}")
for class_name, count in object_counts.items():
    print(f"Total '{class_name}' images extracted: {count}")