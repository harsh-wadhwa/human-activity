# human-activity

In this project LRCN model is used for human activity recognition. 

## Tools and Libraries ##

OpenCV is used to process the video frames.

Scikit-learn is used to divide the dataset into train and test set.

Tensorflow is used to develop and train the model. To train the model almost 40 epochs were run. There is also a fetaure of early stopping callback which terminates
the training of the model if a certain criteria is not met i.e. if the validation loss doesn't improve for continuous 10 epochs then 
the model will stop training.

Stream is used to host the web application.

# UCF50 Dataset

The dataset used is *UCF50 action recognition dataset*. It consist of 50 action categories and each category contains around 25 videos
and it has real videos from youtube.

It performs well in real world setting.
