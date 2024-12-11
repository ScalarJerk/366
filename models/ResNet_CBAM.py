import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from models.CBAM import CBAM


# --- Define ResNet Feature Extractor ---
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        # Load a pretrained ResNet-50 model
        self.resnet = models.resnet50(weights="IMAGENET1K_V1") 
        # Remove the final fully connected layer (classifier)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Extract features till before avgpool

    def forward(self, x):
        x = self.resnet(x)  # Output shape will be (batch_size, c, h, w)
        return x

class CBAMClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CBAMClassifier, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor(pretrained)
        self.cbam = CBAM(channels=2048, reduction=16)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x).view(x.size(0), -1)
        return self.fc(x)
    
'''
# --- Function to Apply CBAM on Features ---
def apply_cbam_to_features(features):
    channels = features.size(1) #gets no. of channels
    cbam = CBAM(channels=channels, reduction=16)
    enhanced_features = cbam(features)
    return enhanced_features

# --- Function to Load and Preprocess the Image ---
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming depth images are grayscale
    #Resize
    image = cv2.resize(image, (224, 224))
    # Convert image to 3D tensor (1, 224, 224) -> (batch_size=1, channels=1, height=224, width=224)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 224, 224]
    # Repeat the grayscale image across 3 channels to match ResNet input shape
    image_tensor = image_tensor.repeat(1, 3, 1, 1)  # Shape: [1, 3, 224, 224]
    # Normalize the image (as ResNet expects)
    image_tensor = image_tensor / 255.0  # Normalize to [0, 1] range
    return image_tensor

# --- Main Function to Extract Features and Apply CBAM ---
def extract_and_enhance_features(image_path):
    # Step 1: Load and preprocess the image
    image_tensor = load_and_preprocess_image(image_path)

    # Step 2: Initialize ResNet feature extractor
    resnet_extractor = ResNetFeatureExtractor()

    # Step 3: Extract features using ResNet
    #with torch.no_grad():  # No need to calculate gradients for feature extraction
    features = resnet_extractor(image_tensor)  # Shape: (1, 2048, h, w)
    print("Extracted feature shape:", features.shape)
    print(features)
    
    # Step 4: Apply CBAM to enhance the features
    enhanced_features = apply_cbam_to_features(features) 
    print("Enhanced feature shape after CBAM:", enhanced_features.shape)
    print(enhanced_features)
    
    # Step 5: Optionally, apply global average pooling to get feature vector of size [1, 2048]
    pooled_features = nn.AdaptiveAvgPool2d((1, 1))(enhanced_features)
    feature_vector = pooled_features.view(pooled_features.size(0), -1)  # Flatten to [batch_size, 2048]

    print("Final feature vector shape after CBAM:", feature_vector.shape)
    return feature_vector

image_path = '/Users/sakshiarjun/Desktop/366/DepthImages2-3/215_18_0_1_1_stand/8028/8028.png'
final_feature_vector = extract_and_enhance_features(image_path)

print("Final feature vector (after CBAM):", final_feature_vector)
'''