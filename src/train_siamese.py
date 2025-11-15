import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

from siamese_network import SiameseNetwork
from dataset import SiameseTripletDataset
from loss import TripletLoss

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED_DATA_DIR = '../data/processed_rescue_data/'

# --- ĐƯỜNG DẪN MỚI ---
# 1. Đường dẫn đến model đã train trên VisDrone (backbone)
VISDRONE_BACKBONE_PATH = '../pretrained_models/best_model.pth' # <-- SỬA TÊN FILE NÀY CHO ĐÚNG
# 2. Tên file model Siamese cuối cùng sẽ được lưu
MODEL_SAVE_PATH = '../pretrained_models/best_siamese_model_from_visdrone.pth' 

# Siêu tham số cho training
IMG_SIZE = 128
EPOCHS = 50
BATCH_SIZE = 32
# QUAN TRỌNG: Learning rate NHỎ vì đây là fine-tuning
LR = 1e-4      
GAMMA = 0.7    
MARGIN = 1.0   
NUM_WORKERS = 4 if os.name == 'posix' else 0

# --- 1. Data Augmentation và DataLoader ---
# (Phần này giữ nguyên, không cần thay đổi)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("--- Preparing Datasets ---")
full_dataset = SiameseTripletDataset(data_dir=PROCESSED_DATA_DIR, transform=train_transform)
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- 2. Model, Loss, Optimizer, Scheduler ---
print(f"--- Using device: {DEVICE} ---")
print("--- Initializing Model for Fine-tuning ---")

# --- THAY ĐỔI CÁCH KHỞI TẠO MODEL ---
# Truyền đường dẫn của VisDrone backbone vào đây
model = SiameseNetwork(backbone_path=VISDRONE_BACKBONE_PATH, freeze_backbone=False).to(DEVICE)

criterion = TripletLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=10, gamma=GAMMA)

# --- 3. Vòng lặp Training & Validation ---
# (Phần này giữ nguyên, không cần thay đổi)
best_val_loss = float('inf')
print("--- Starting Fine-tuning on Rescue Dataset ---")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for anchor, positive, negative in pbar:
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
        
        optimizer.zero_grad()
        output_anchor, output_positive, output_negative = model(anchor, positive, negative)
        loss = criterion(output_anchor, output_positive, output_negative)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for anchor, positive, negative in pbar_val:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            output_anchor, output_positive, output_negative = model(anchor, positive, negative)
            loss = criterion(output_anchor, output_positive, output_negative)
            val_loss += loss.item()
            pbar_val.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"*** New best model saved to {MODEL_SAVE_PATH} with val loss: {best_val_loss:.4f} ***")

print("--- Training finished ---")
print(f"Best model saved at: {MODEL_SAVE_PATH}")