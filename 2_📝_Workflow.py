import streamlit as st
from PIL import Image

st.header("📝 Workflow")
st.write("The process of developing a robust system for detection and quantification of pavement distresses is divided into five stages:")
st.write("1. Pre-processing: The pre-processing stage includes preparing the dataset for further analysis by performing image resizing, and normalization, to ensure that the images are of the same quality and size.")
st.write("2. Pavement Distress Classification: This stage involves the classification of the images as either having or not having pavement distresses. The classification process is performed using Convolutional Neural Networks (CNN), which can learn from a large dataset of labeled images to identify patterns and features associated with pavement distresses.")
st.write("3. Pavement Distress Segmentation: Segmentation stage involves finding the regions of interest for crack and potholes in the given image. This Segmentation task is performed using Deep Learning techniques such as YOLO-v5 and U-Net architecture.")
st.write("4. Pavement Distress Quantification: Once the pavement distresses are classified and segmented, the next step is to quantify the extent and severity of the distress. The quantification process involves measuring the length, and area of the distresses.")
st.write("5. Web-Application: The final stage involves the development of a web-based application that allows the users to visualize and analyze the pavement distress.")


workflow = Image.open('workflow.png')
st.image(workflow, caption='Methodology Flowchart')