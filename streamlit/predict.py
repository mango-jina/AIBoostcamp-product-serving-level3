import torch
import streamlit as st
from model import *
from utils import transform_image
from typing import Tuple

@st.cache
def load_model() -> UnetPlusPlus_Efficient5:
    model_path = 'model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UnetPlusPlus_Efficient5().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def get_prediction(model:UnetPlusPlus_Efficient5, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)
    oms = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    return tensor, oms
