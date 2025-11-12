import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from siamese_network import SiameseNetwork
from dataset import SiameseTripletDataset
from loss import TripletLoss
from tqdm import tqdm

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED_DATA_DIR = '../data/processed_rescue_data/'
BACKBONE_PATH = '../pre-trained_models/best_model.pth'
MODEL_SAVE_PATH = '../pre-trained_models/best_siamese_model.pth'

IMG_SIZE = 128
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.0005
GAMMA = 0.7 # Tỉ lệ giảm LR của scheduler
MARGIN = 1.0 # Margin cho Triplet Loss

# --- 1. Data Augmentation và DataLoader ---
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

full_dataset = SiameseTripletDataset(data_dir=PROCESSED_DATA_DIR, transform=train_transform)
# Note: ideally, val_dataset should use val_transform, but for simplicity we use one dataset object.
# For a more rigorous setup, create two dataset objects with different transforms.

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- 2. Model, Loss, Optimizer, Scheduler ---
model = SiameseNetwork(backbone_path=BACKBONE_PATH, freeze_backbone=False).to(DEVICE)
criterion = TripletLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=10, gamma=GAMMA)

# --- 3. Vòng lặp Training & Validation ---
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # Training phase
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
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
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

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"*** New best model saved with val loss: {best_val_loss:.4f} ***")

print("--- Training finished ---")