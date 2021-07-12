import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import base64
from io import BytesIO
from PIL import Image

## Main web-app using Streamlit

faceCascade = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')

## Video Processing function
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        i =self.i+1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            cv2.putText(img, 'F-' + str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img

## Generate download link
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

## Detect face
def face_detect(image,sf,mn):
    i = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,sf,mn)
    for (x, y, w, h) in faces:
        i = i+1
        cv2.rectangle(image, (x, y), (x + w, y + h), (237, 30, 72), 3)
        cv2.rectangle(image, (x, y - 40), (x + w, y),(237, 30, 72) , -1)
        cv2.putText(image, 'F-'+str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    resi_image = cv2.resize(image, (350, 350))
    return resi_image,i,image

## UI for the App
def run():
    st.title("Face Detection using OpenCV")
    activities = ["Image", "Webcam"]
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Choose Input Source")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '[©Developed by Spidy20](http://github.com/spidy20)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    if choice == 'Image':
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Face Detection is done using Haar Cascade & OpenCV"</h4>''',
            unsafe_allow_html=True)
        img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])
        if img_file is not None:
            img = np.array(Image.open(img_file))
            img1 = cv2.resize(img, (350, 350))
            place_h = st.beta_columns(2)
            place_h[0].image(img1)
            st.markdown(
                '''<h4 style='text-align: left; color: #d73b5c;'>* Increase & Decrease it to get better accuracy.</h4>''',
                unsafe_allow_html=True)
            scale_factor = st.slider("Set Scale Factor Value", min_value=1.1, max_value=1.9, step=0.10, value=1.3)
            min_Neighbors = st.slider("Set Scale Min Neighbors", min_value=1, max_value=9, step=1, value=5)
            fd, count, orignal_image = face_detect(img, scale_factor, min_Neighbors)
            place_h[1].image(fd)
            if count == 0:
                st.error("No People found!!")
            else:
                st.success("Total number of People : " + str(count))
                result = Image.fromarray(orignal_image)
                st.markdown(get_image_download_link(result, img_file.name, 'Download Image'), unsafe_allow_html=True)
    if choice == 'Webcam':
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* It might be not work with Android Camera"</h4>''',
            unsafe_allow_html=True)
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
run()