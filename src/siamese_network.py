import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    """
    Kiến trúc mạng Siamese sử dụng backbone MobileNetV3-Large.
    Tự động tải trọng số pre-trained từ ImageNet.
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        Khởi tạo mạng.
        :param pretrained: bool - Nếu True, tải trọng số pre-trained trên ImageNet.
        :param freeze_backbone: bool - Nếu True, đóng băng các lớp của backbone và chỉ train lớp classifier mới.
        """
        super(SiameseNetwork, self).__init__()
        
        # 1. Tải kiến trúc MobileNetV3-Large với trọng số pre-trained trên ImageNet
        print(f"Loading MobileNetV3-Large backbone with pretrained weights from ImageNet...")
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # 2. Thay thế lớp classifier cuối cùng để tạo ra vector embedding
        # QUAN TRỌNG: MobileNetV3 có cấu trúc classifier khác V2.
        # Lấy số features từ lớp Linear ĐẦU TIÊN của classifier gốc.
        num_ftrs = self.backbone.classifier[0].in_features
        
        # Tạo một classifier mới của riêng chúng ta
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128) # Vector embedding 128 chiều
        )

        print("Successfully loaded pretrained MobileNetV3 backbone and replaced classifier head.")
            
        # 3. (Tùy chọn) Đóng băng các lớp của backbone để fine-tune
        if freeze_backbone:
            print("Freezing backbone layers. Only the new classifier head will be trained.")
            for name, param in self.backbone.named_parameters():
                # Chỉ đóng băng các lớp không phải là 'classifier'
                if 'classifier' not in name:
                    param.requires_grad = False
        else:
            print("All layers are trainable (full fine-tuning).")

    def forward_one(self, x):
        """
        Chạy một ảnh qua backbone để lấy embedding.
        Hàm này sẽ được sử dụng trong quá trình inference (phát hiện).
        """
        return self.backbone(x)

    def forward(self, anchor, positive, negative):
        """
        Chạy cả ba ảnh (anchor, positive, negative) qua mạng.
        Hàm này được sử dụng trong quá trình training với Triplet Loss.
        """
        output_anchor = self.forward_one(anchor)
        output_positive = self.forward_one(positive)
        output_negative = self.forward_one(negative)
        return output_anchor, output_positive, output_negative