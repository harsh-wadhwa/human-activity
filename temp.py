# # from keras.models import load_model
from backend import *
# # from keras.models import load_model
# # from backend import *
# # CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRiding"]

# # model = load_model("C:/rnd_project/LRCN_model___Date_Time_2022_05_04__10_13_56___Loss_0.37244558334350586___Accuracy_0.9180327653884888.h5")
# # # predict_single_action("C:/rnd_project/Test Video.mp4",20)
# # predict_on_video("C:/rnd_project/Test Video.mp4","C:/rnd_project/Test Video-out.mp4",20)
# import os
# list = os.listdir(r"G:\My Drive\RnD\UCF50")
# print(list)

vid_title = download_youtube_videos("https://www.youtube.com/watch?v=lq2REad-o0Y","C:/rnd_project/")
print(vid_title+".mp44")