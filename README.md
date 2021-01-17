# Anti-Candid_Pose
Anti-Candid_Pose

1.
source /home/minggatsby/python/bin/activate

2.
cd src/Anti-Candid/



3.
Object Detection

If it use gstreamer
python3 trt_yolo.py -m yolov4-416 --usb 0
If it doesn't use gstreamer
python3 trt_yolo.py -m yolov4-416 --usb 0 --gstreamer false 


Pose_Estimation

cd src/Anti-Candid/trt_pose/tasks/human_pose/

python3 start.py


Object Detection & Pose_Estimation

python3 detect_camera.py --usb 0 -g 
