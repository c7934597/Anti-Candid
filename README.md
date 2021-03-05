# Anti-Candid_Pose
Anti-Candid_Pose

## 1. Getting Started
(可選)
Replace the OSD binaries (x86 or Jetson) in $DEEPSTREAM_DIR/libs with the ones provided in this repository under bin/. Please note that these are not inter-compatible across platforms.

## 2. Path
cd /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-app-yolo-and-pose/

## 3. Build
先複製 /opt/nvidia/deepstream/deepstream-5.1/sources/apps/apps-common 到 /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/ 底下，然後貼上Code的資料夾

(x86)
sudo apt-get install libgstreamer1.0-dev

(x86)
sudo apt-get install libgstreamer-plugins-base1.0-dev

sudo apt-get install libjson-glib-dev

sudo apt-get install libgstrtspserver-1.0-dev

CUDA_VER=10.2 make

CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo

(If memory isn't enough, it must generate engine alone)
/usr/src/tensorrt/bin/trtexec --onnx=pose_estimation.onnx --saveEngine=pose_estimation.onnx_b1_gpu0_fp16.engine

## 4. Run
./deepstream-app -c deepstream_app_config.txt


# References
https://spyjetson.blogspot.com/

https://github.com/jkjung-avt/tensorrt_demos

