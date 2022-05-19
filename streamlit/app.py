import streamlit as st

import io
import time

from PIL import Image
import matplotlib.pyplot as plt

from predict import load_model, get_prediction
from visualization import *

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'


def main():
    st.markdown("---")
    st.markdown("<h1 style='text-align: center'>Trash Segmentation</h1>", unsafe_allow_html=True)
    st.write('')
    
    # model load
    st.header("1. Load a Model")
    st.write('')
    try:
        model = load_model()
        st.write(f"**** Model is <span style='color:red'>{model.__class__.__name__}</span> ****", unsafe_allow_html=True)
        st.success("Load Success!")
    except Exception as err: 
        st.error(f"Error message: {err}")
    
    model.eval()
    st.write('')
    
    # image upload
    st.header("2. Upload Images")
    st.write('')
    uploaded_file = st.file_uploader("Choose Images (limited to 200MB)", accept_multiple_files=True, type=["jpg", "jpeg","png"])
    
    # model inference
    try:
        for idx, img_file in enumerate(uploaded_file):
            if idx == 0:
                with st.spinner("Prediction..."):
                    time.sleep(2)
                st.write('')
                st.header("3. Inference")
                st.write('')
    
            if img_file:
                st.markdown(f'#### #{idx+1}. {img_file.name}')
                image_bytes = img_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))

                st.image(image, caption='Original Image', width=512)
                
                _, oms = get_prediction(model, image_bytes)
                mask = label_to_color_image(oms)
                
                class_colormap = pd.read_csv("class_dict.csv")
                category_and_rgb = [[category, (r,g,b)] for id, (category, r, g, b) in enumerate(class_colormap.values)]
                legend_elements = [Patch(facecolor=webcolors.rgb_to_hex(rgb), 
                                    edgecolor=webcolors.rgb_to_hex(rgb), 
                                    label=category) for category, rgb in category_and_rgb]
                f, ax = plt.subplots(figsize=(6.6, 6.6))
                ax.imshow(mask)
                ax.axis("off")
                ax.legend(frameon=False, handles=legend_elements, bbox_to_anchor=(0,0), loc=2, ncol=4, borderaxespad=0, prop={'size':10})
                
                buf = io.BytesIO()
                f.savefig(buf, format="png", bbox_inches='tight', pad_inches = 0)
                st.image(buf, caption='Segmentation Result')
            
            if idx == len(uploaded_file)-1:
                st.success("Segmentation Complete!")
            
    except Exception as err: 
        st.error(f"Error message: {err}")


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')
