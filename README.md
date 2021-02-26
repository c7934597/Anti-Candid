# Anti-Candid_Pose
Anti-Candid_Pose

## 1. Getting Started
Replace the OSD binaries (x86 or Jetson) in $DEEPSTREAM_DIR/libs with the ones provided in this repository under bin/. Please note that these are not inter-compatible across platforms.

## 2. Path
cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream_yolo_and_pose_estimation/

## 3. Build
sudo apt-get install libgstreamer1.0-dev

sudo apt-get install libgstreamer-plugins-base1.0-dev

sudo apt-get install libjson-glib-dev

sudo apt-get install libgstrtspserver-1.0-dev

make

CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo

## 4. Run
./deepstream-app -c deepstream_app_config.txt


# References
https://spyjetson.blogspot.com/

https://github.com/jkjung-avt/tensorrt_demos

