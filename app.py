from statistics import mode
import streamlit as st
import pickle
from keras.models import load_model
from backend import *
st.title("Human Activity Detection")
# 1. manual upload
file_vid = st.file_uploader(label="Upload a file")
# print(file_vid[0])
# model_2 = load_model("C:/rnd_project/LRCN_model___Date_Time_2022_05_04__10_13_56___Loss_0.37244558334350586___Accuracy_0.9180327653884888.h5")
model = load_model("C:/rnd_project/LRCN_model___Date_Time_2022_05_28__09_16_25___Loss_0.6534330248832703___Accuracy_0.8526119589805603.h5")

output_file = "C:/rnd_project/output.mp4"
if file_vid is not None: 
    predict_on_video(file_vid.name,output_file,20)
    video_file = open(output_file, "rb").read()
    st.video(video_file)


# 2. youtube link
yt_link = st.text_input("Enter youtube link:")
if yt_link is not None:
    vid_title = download_youtube_videos(yt_link,"C:/rnd_project/")
    vid = vid_title+".mp4"
    predict_on_video("C:/rnd_project/Super lady ## HORSE Riding ## horse game ## Fast Horse riding.mp4",output_file,20)
    video_file = open(output_file, "rb").read()
    st.video(video_file)