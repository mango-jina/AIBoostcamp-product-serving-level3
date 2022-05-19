import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

class UnetPlusPlus_Efficient5(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "timm-efficientnet-b5"
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name, 
            encoder_weights="noisy-student",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='noisy-student')
    
    def forward(self, x):
        return self.model(x)    