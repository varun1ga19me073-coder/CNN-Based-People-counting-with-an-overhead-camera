# CNN-Based-People-counting-with-an-overhead-camera
This is the code for my Master's thesis at SRH University of Applied Sciences Berlin (Engineering and Sustainable Technology Management). The thesis is called CNN-based passenger counting with an overhead camera.

The idea is simple: a camera looks down from above, and a small neural network counts how many people walk across a line in each direction. What I want to find out is how well this works with lightweight YOLO models, and how much accuracy is lost when the whole thing runs on a small device like a Raspberry Pi 5.

Where things stand: this is a work in progress. The counting pipeline runs from start to finish on test videos. I have downloaded a public overhead dataset, but I have not trained on it yet, and the Raspberry Pi tests are still ahead of me. I will add results here as I get them.

What I am trying to find out
Which of two or three small YOLO models (YOLOv8n, YOLO11n, maybe YOLOv8s) counts best when they all get the same data and the same training settings.
How far the program's counts are from counts I make by hand.
Where it goes wrong: groups of people, bags, crowds, people who stop right on the line.
How the chosen model behaves on a Raspberry Pi 5, and how much accuracy and speed change compared with my laptop.
How the counter works

It does three things for every frame of the video:

YOLO finds the people and puts a box around each one.
A tracker (ByteTrack) gives each person an ID, so the same person is followed from frame to frame.
A line is drawn across the image. Whenever a tracked person moves from one side of the line to the other, the program counts one IN or one OUT, depending on the direction. There is a small dead zone around the line, so someone standing on it is not counted twice.
What is in this repository
count_people.py runs the counter on a video and saves an annotated video plus a CSV log of every crossing.
train.py fine-tunes a YOLO model on an overhead dataset and prints precision, recall and mAP50 at the end.
extract_frames.py cuts a video into pictures. I only need it if I label my own footage, which I do not plan to do for now.
overhead.yaml tells the training script where the dataset is and what the class is called.
Getting started

You need Python 3.10 or newer. Then install the two libraries:

Terminal
pip install ultralytics opencv-python

If you are on Windows and pip fails with an error about long paths, install into a short virtual environment instead. This fixed it for me:

Terminal
python -m venv C:\v
C:\v\Scripts\activate
pip install ultralytics opencv-python
Try the counter first
Terminal
python count_people.py --source people.mp4 --weights yolov8n.pt --classes 0

When it finishes you get counted.mp4 (the video with boxes, the line and the IN / OUT numbers) and crossings.csv. If the video will not play, open it with VLC.

A few options are worth knowing:

--line 0.75 moves the line. It is a fraction of the frame, so 0.5 is the middle. Put it where people really cross.
--axis x uses a vertical line, for people who walk left to right in the picture.
--conf 0.25 makes the detector less strict, which helps when people are small.
--flip swaps IN and OUT if they come out the wrong way round.
--show opens a live window while it runs. Press q to close it.

A heads-up from my own first test: a very crowded clip with tiny people gave me zero crossings, because the line sat where almost nobody walked. A clearer clip and a better line position matter a lot. That crowded clip is also a good example for the error analysis later.

Get an overhead dataset

The pretrained model was mostly trained on photos taken from the side, so it is weak on top-down views. I use public overhead data, for example the Roboflow Universe dataset "topview people detection" (CC BY 4.0). Download it in YOLOv8 format, unzip it into a folder called dataset next to the scripts, and check how the folders are named. Roboflow usually gives you something like this:

dataset/
    train/images, train/labels
    valid/images, valid/labels
    data.yaml

In that case overhead.yaml should say:

yaml
path: dataset
train: train/images
val: valid/images

names:
  0: person

Look at the dataset's own data.yaml to see what its class is called, and make sure there is only one class.

Train
Terminal
python train.py --data overhead.yaml --model yolov8n.pt --epochs 50

The best model ends up in runs/detect/overhead_people/weights/best.pt. Training on a laptop without a GPU is slow, so Google Colab is an option for public data. To train another model under the same conditions, change --model (for example yolo11n.pt) and give it a different --name so the results do not overwrite each other.

Then run the counter again with your own model:

Terminal
python count_people.py --source test_video.mp4 --weights runs/detect/overhead_people/weights/best.pt
Check how good it is

Watch each test video and count the people crossing the line yourself. Then compare with the program:

error (%) = |counted - actual| / actual * 100

I report this per video and per model, next to the detection scores from train.py.

Data and privacy

I only use public datasets for now, and nothing showing real people is stored in this repository. If I ever record my own footage, it will only be with the consent of the people in it, and it will not be uploaded here. The .gitignore keeps videos, datasets and model files out of the repository, and the counting code saves only counts and crossing events, never images.

Results

Nothing to show yet. Once the experiments are done, I will write the numbers here for each model on the laptop and for the best one on the Raspberry Pi 5: precision, recall, mAP50, counting error, time per image and memory use.

Credits
Ultralytics YOLO (AGPL-3.0), including the ByteTrack tracker.
The "topview people detection" dataset on Roboflow Universe (CC BY 4.0). Please credit the dataset authors as the license asks.
Author

Varun Kumar Gowda Gangadhara
Supervisor: Prof. Dr. Oezdemir Cetin
