import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_path=None, pretrained_torch=False, freeze_backbone=False):
        super(SiameseNetwork, self).__init__()
        
        # Quyết định tải backbone nào
        if backbone_path:
            print(f"Loading MobileNetV3-Large backbone and fine-tuning from: {backbone_path}")
            # Tải kiến trúc không có trọng số pre-trained từ torchvision
            self.backbone = models.mobilenet_v3_large(weights=None)
        elif pretrained_torch:
            print(f"Loading MobileNetV3-Large backbone with pretrained weights from ImageNet...")
            self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        else:
            print("Initializing MobileNetV3-Large backbone with random weights.")
            self.backbone = models.mobilenet_v3_large(weights=None)
            
        # Thay thế lớp classifier
        num_ftrs = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

        # Nếu có đường dẫn file, tải trọng số từ đó
        if backbone_path:
            try:
                state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))
                # Xử lý các trường hợp key có prefix 'module.'
                if all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # strict=False để bỏ qua các key không khớp (ví dụ: classifier cũ)
                self.backbone.load_state_dict(state_dict, strict=False)
                print("Backbone weights from VisDrone model loaded successfully!")
            except Exception as e:
                print(f"Could not load weights from {backbone_path}: {e}")

        if freeze_backbone:
            # ... (phần đóng băng giữ nguyên)
            pass

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_one(anchor)
        output_positive = self.forward_one(positive)
        output_negative = self.forward_one(negative)
        return output_anchor, output_positive, output_negative