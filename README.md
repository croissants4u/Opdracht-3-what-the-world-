# Opdracht-3-what-the-world-
Opdracht 3: What the world?!

// Code for running yolov5 seperately off camera source (source default set to "0") //
python detect.py --source 0
python detect.py --weights yolov5x.pt --source 0

// Code for running yolov5 seperately off video (mp4) source (source default set to ~Desktop/*yourvideo*.mp4) //
python detect.py --weights yolov5x.pt --data ~/Desktop/*yourvideo.mp4* --view-img

// To start the image generation/yolov5 just run the code below in the terminal //
// make sure you only have 1 camera plugged in or that the camera you want to use is set to source "0"//
// Add your openai api key on the 11 line of the start script to run the code properly //
python StartScript.py
