import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_path, freeze_backbone=False):
        super(SiameseNetwork, self).__init__()
        
        self.backbone = models.mobilenet_v2(pretrained=False)
        
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512), # Thêm BatchNorm để ổn định training
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

        print(f"Loading backbone weights from: {backbone_path}")
        try:
            state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))
            # Xử lý các trường hợp key có prefix 'module.' (khi train bằng DataParallel)
            if all(key.startswith('module.') for key in state_dict.keys()):
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.backbone.load_state_dict(state_dict, strict=False)
            print("Backbone weights loaded successfully!")
        except Exception as e:
            print(f"Could not load pre-trained weights: {e}. Training from scratch or ImageNet.")
            
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_one(anchor)
        output_positive = self.forward_one(positive)
        output_negative = self.forward_one(negative)
        return output_anchor, output_positive, output_negative