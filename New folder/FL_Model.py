"""
Convolutional Neural Network Architecture for Federated Medical Diagnosis Using X-Ray Images.
Project: Machine Learning Based on Federated Medical Diagnosis Using X-Ray Images
"""

import torch
import torch.nn as nn

class MedicalXRayCNN(nn.Module):
    """
    Deep Convolutional Neural Network for Medical X-Ray Image Diagnosis.
    Performs binary medical classification: Normal (Healthy) vs Pathological (Disease Detected).
    Input image dimension: 128 x 128 x 3 (or expanded 1-channel grayscale radiograph).
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        # ConvBlock 1: Extract low-level spatial radiographic features (edges, anatomical boundaries)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # ConvBlock 2: Extract intermediate textures & radiodensity variations
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # ConvBlock 3: Extract high-level pathological lesion & diagnostic patterns
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Spatial adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Diagnostic Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Backward compatibility aliases
OsteoXRayCNN = MedicalXRayCNN
CloudSecurityModel = MedicalXRayCNN
