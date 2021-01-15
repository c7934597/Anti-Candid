# Anti-Candid_Pose_Estimation_And_Object_Detection
Anti-Candid_Pose_Estimation_And_Object_Detection

1.
source /home/minggatsby/python/bin/activate

2.
cd src/Anti-Candid_Pose_Estimation_And_Object_Detection/



3.
Object Detection

If it use gstreamer
python3 trt_yolo.py -m yolov4-416 --usb 0
If it doesn't use gstreamer
python3 trt_yolo.py -m yolov4-416 --usb 0 --gstreamer false 


Pose_Estimation

cd src/Anti-Candid_Pose_Estimation_And_Object_Detection/trt_pose/tasks/human_pose/

python3 detect_camera.py
