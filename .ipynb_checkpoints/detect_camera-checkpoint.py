import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np

import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path

"""Parse input arguments."""
parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
parser.add_argument('--video', type=str, default='/home/minggatsby/src/test_images/video.avi', help = 'video file name' )
args = parser.parse_args()

with open('./trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = './trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = './trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = './trt_pose/tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = './trt_pose/tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    print('-- Converting TensorRT models. This may takes several minutes...')
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def draw_keypoints(img, key):
    thickness = 5
    w, h = img.size
    draw = PIL.ImageDraw.Draw(img)
    #draw Rankle -> RKnee (16-> 14)
    if all(key[16]) and all(key[14]):
        draw.line([ round(key[16][2] * w), round(key[16][1] * h), round(key[14][2] * w), round(key[14][1] * h)],width = thickness, fill=(51,51,204))
    #draw RKnee -> Rhip (14-> 12)
    if all(key[14]) and all(key[12]):
        draw.line([ round(key[14][2] * w), round(key[14][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=(51,51,204))
    #draw Rhip -> Lhip (12-> 11)
    if all(key[12]) and all(key[11]):
        draw.line([ round(key[12][2] * w), round(key[12][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lhip -> Lknee (11-> 13)
    if all(key[11]) and all(key[13]):
        draw.line([ round(key[11][2] * w), round(key[11][1] * h), round(key[13][2] * w), round(key[13][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lknee -> Lankle (13-> 15)
    if all(key[13]) and all(key[15]):
        draw.line([ round(key[13][2] * w), round(key[13][1] * h), round(key[15][2] * w), round(key[15][1] * h)],width = thickness, fill=(51,51,204))

    #draw Rwrist -> Relbow (10-> 8)
    if all(key[10]) and all(key[8]):
        draw.line([ round(key[10][2] * w), round(key[10][1] * h), round(key[8][2] * w), round(key[8][1] * h)],width = thickness, fill=(255,255,51))
    #draw Relbow -> Rshoulder (8-> 6)
    if all(key[8]) and all(key[6]):
        draw.line([ round(key[8][2] * w), round(key[8][1] * h), round(key[6][2] * w), round(key[6][1] * h)],width = thickness, fill=(255,255,51))
    #draw Rshoulder -> Lshoulder (6-> 5)
    if all(key[6]) and all(key[5]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[5][2] * w), round(key[5][1] * h)],width = thickness, fill=(255,255,0))
    #draw Lshoulder -> Lelbow (5-> 7)
    if all(key[5]) and all(key[7]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[7][2] * w), round(key[7][1] * h)],width = thickness, fill=(51,255,51))
    #draw Lelbow -> Lwrist (7-> 9)
    if all(key[7]) and all(key[9]):
        draw.line([ round(key[7][2] * w), round(key[7][1] * h), round(key[9][2] * w), round(key[9][1] * h)],width = thickness, fill=(51,255,51))

    #draw Rshoulder -> RHip (6-> 12)
    if all(key[6]) and all(key[12]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=(153,0,51))
    #draw Lshoulder -> LHip (5-> 11)
    if all(key[5]) and all(key[11]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(153,0,51))


    #draw nose -> Reye (0-> 2)
    if all(key[0][1:]) and all(key[2]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[2][2] * w), round(key[2][1] * h)],width = thickness, fill=(219,0,219))

    #draw Reye -> Rear (2-> 4)
    if all(key[2]) and all(key[4]):
        draw.line([ round(key[2][2] * w), round(key[2][1] * h), round(key[4][2] * w), round(key[4][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> Leye (0-> 1)
    if all(key[0][1:]) and all(key[1]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[1][2] * w), round(key[1][1] * h)],width = thickness, fill=(219,0,219))

    #draw Leye -> Lear (1-> 3)
    if all(key[1]) and all(key[3]):
        draw.line([ round(key[1][2] * w), round(key[1][1] * h), round(key[3][2] * w), round(key[3][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> neck (0-> 17)
    if all(key[0][1:]) and all(key[17]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[17][2] * w), round(key[17][1] * h)],width = thickness, fill=(255,255,0))
    return img

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
    
'''
Draw to original image
'''
def execute(img, org, count):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        #print("Human index:%d "%( i ))
        kpoint = get_keypoint(objects, i, peaks)
        #print(kpoint)
        org = draw_keypoints(org, kpoint)
    netfps = 1 / (end - start)  
    draw = PIL.ImageDraw.Draw(org)
    draw.text((30, 30), "NET FPS:%4.1f"%netfps, font=fnt, fill=(0,255,0))    
    print("Human count:%d len:%d "%(counts[0], len(counts)))
    print('===== Frmae[%d] Net FPS :%f ====='%(count, netfps))
    return org

'''
v4l2-ctl --list-formats-ext指令查詢規範
sudo systemctl restart nvargus-daemon指令重啟相機服務
'''
_WIDTH=640
_HEIGHT=480
_FLIP_METHOD=0

# 選擇相機 MIPI / USB
# camSet = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1, format=NV12 ! nvvidconv flip-method='+str(_FLIP_METHOD)+' ! video/x-raw, width='+str(_WIDTH)+', height='+str(_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# camSet = 'v4l2src device=/dev/video0 ! video/x-raw, width='+str(_WIDTH)+', height='+str(_HEIGHT)+', framerate=24/1 ! videoconvert ! appsink'

# cap = cv2.VideoCapture(camSet)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(args.video)

ret_val, img = cap.read()
# H, W, __ = img.shape

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# dir, filename = os.path.split(args.video)
# name, ext = os.path.splitext(filename)
# out_video = cv2.VideoWriter('/home/minggatsby/src/test_images/result/%s_%s.mp4'%(args.model, name), fourcc, cap.get(cv2.CAP_PROP_FPS), (W, H))
# count = 0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

fontname = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fnt = PIL.ImageFont.truetype(fontname, 45)

count = 1
while cap.isOpened():
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Frame Read End")
        break       
    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
#     pilimg = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
#     pilimg = PIL.Image.fromarray(pilimg)
    pilimg = PIL.Image.fromarray(dst)
    pilimg = execute(img, pilimg, count)
    array = np.asarray(pilimg, dtype="uint8")
    # out_video.write(array)
    count += 1

    # 顯示影像
#     cv2.imshow('myCam', dst) # 顯示原始影像
    cv2.imshow('myCam', array) # 顯示姿態估計影像

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
# out_video.release()
cap.release()