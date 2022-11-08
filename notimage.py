import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import json
import os


count =0

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Specify the path to model config and checkpoint file
config_file = 'configs/yolof/yolof_r50_c5_8x8_1x_coco.py'
checkpoint_file = './epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

#for i in range(38875):
final_result=[]

# test a single image and show the results
#name = "20210906_RGB_H_005_90_02"
source = "../ai_plant_data/traindata/data/Validation/"
#choice_label = "label/VL14/14/"


#img1 = "../val/"+name+ ".jpg"
#print(img)
#label = source + choice_label + name+ ".json"
#print(label)
img_dir= "../val"
mean = float(0)
print("read val dir")

file_list = os.listdir(img_dir)
print(len(file_list))


for i in range(len(file_list)):
    name = file_list[i]
    root = name[13]

    if root == "A":
        choice_label = "label/VL1/1/"
        
    elif root == "L":
        choice_label = "label/VL2/2/"
        pass
    elif root == "M":
        choice_label = "label/VL3/3/"
        pass
    elif root == "G":
        choice_label = "label/VL4/4/"
        pass
    elif root == "K":
        choice_label = "label/VL5/5/"
        pass
    elif root == "B":
        choice_label = "label/VL6/6/"
        pass
    elif root == "N":
        choice_label = "label/VL7/7/"
        pass
    elif root == "E":
        choice_label = "label/VL8/8/"
        pass
    elif root == "F":
        choice_label = "label/VL9/9/"
        pass
    elif root == "C":
        choice_label = "label/VL10/10/"
        pass
    elif root == "D":
        choice_label = "label/VL11/11/"
        pass
    elif root == "J":
        choice_label = "label/VL12/12/"
        pass
    elif root == "I":
        choice_label = "label/VL13/13/"
        pass
    elif root == "H":
        choice_label = "label/VL14/14/"
        pass

    name = name[:-4]
    img1 = "../val/"+name + ".jpg"
    label = source + choice_label + name+ ".json"



    with open(label, 'r') as f:
        json_data = json.load(f)
    bbox = json_data['annotations']['bbox'] #['annotations']['bbox']
    gt_label = json_data['annotations']['weeds_kind']  #['categories]['name']

    if count == 0:
        if gt_label == "A":
            print("gt_label: 강피") #1
            gt_label = "강피"

        elif gt_label == "L":
            print("gt_label: 개비름") #2
            gt_label = "개비름"

        elif gt_label == "M":
            print("gt_label: 개여뀌") #3
            gt_label = "개여뀌"

        elif gt_label == "G":
            print("gt_label: 깨풀") #4
            gt_label = "깨풀"

        elif gt_label == "K":
            print("gt_label: 닭의장풀") #5
            gt_label = "닭의장풀"

        elif gt_label == "B":
            print("gt_label: 물달개비") #6
            gt_label = "물달개비"

        elif gt_label == "N":
            print("gt_label: 미국가막사리") #7
            gt_label = "미국가막사리"

        elif gt_label == "E":
            print("gt_label: 바랭이") #8 
            gt_label = "바랭이"

        elif gt_label == "F":
            print("gt_label: 쇠비름") #9
            gt_label = "쇠비름"

        elif gt_label == "C": 
            print("gt_label: 올방개") #10
            gt_label = "올방개"

        elif gt_label == "D":
            print("gt_label: 올챙이고랭이") #11
            gt_label = "올챙이고랭이"

        elif gt_label == "J":
            print("gt_label: 좀명아주") #12
            gt_label = "좀명아주"    

        elif gt_label == "I":
            print("gt_label: 한련초") #13
            gt_label = "한련초"

        elif gt_label == "H":
            print("gt_label: 흰명아주") #14
            gt_label = "흰명아주"

    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))


    gt_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    result = inference_detector(model, img1)
    # visualize the results in a new window

    image, bboxes, pred = model.show_result(img1, result, bbox_color=[(0,255,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)], text_color=(255, 255, 255), thickness=25, font_size=42)
    if type(bboxes) != str:
        pt3 = (int(bboxes[0]), int(bboxes[1]))
        pt4 = (int(bboxes[2]), int(bboxes[3]))

        #image = cv2.imread(img1)
        #image = cv2.rectangle(image, pt1, pt2, (0,0,255), 20)
        #image = cv2.rectangle(image, pt3, pt4, (0, 255, 0), 20)
#
#
        #pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #pil_image=Image.fromarray(pil_image)
        #fontpath = "../../usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"
        #font = ImageFont.truetype(fontpath, 250)
        #draw = ImageDraw.Draw(pil_image)
        #fonts = ImageFont.truetype(fontpath, 250)
#
        cal = IoU(gt_bbox, bboxes)

        cal = round(cal, 2)
        #w,h,c =image.shape


        #    CLASSES = ('gangpi', 'gaebireum', 'gaeyeoggui', 'ggaepul', 'jangpul', 'muldalgaebi', 
        # 'gamaksari', 'baraengi', 'soibireum', 'olbanggae', 'goraengi', 'jommyeongaju', 'hanryeoncho', 'heonmyeongaju')

        if count == 0:
            if pred == "gangpi":
                print("prediction: 강피")
                tmp = "강피"

            elif pred == "gaebireum":
                print("prediction: 개비름")
                tmp = "개비름"

            elif pred == "gaeyeoggui":
                print("prediction: 개여뀌")
                tmp = "개여뀌"

            elif pred == "ggaepul":
                print("prediction: 깨풀")
                tmp = "깨풀"

            elif pred == "jangpul":
                print("prediction: 닭의장풀")
                tmp = "닭의장풀"

            elif pred == "muldalgaebi":
                print("prediction: 물달개비")
                tmp = "물달개비"

            elif pred == "gamaksari":
                print("prediction: 미국가막사리")
                tmp = "미국가막사리"

            elif pred == "baraengi":
                print("prediction: 바랭이")
                tmp = "바랭이"

            elif pred == "soibireum":
                print("prediction: 쇠비름")
                tmp = "쇠비름"

            elif pred == "olbanggae":
                print("prediction: 올방개")
                tmp = "올방개"

            elif pred == "goraengi":
                print("prediction: 올챙이고랭이")
                tmp = "올챙이고랭이"

            elif pred == "jommyeongaju":
                print("prediction: 좀명아주")
                tmp = "좀명아주"    

            elif pred == "hanryeoncho":
                print("prediction: 한련초")
                tmp = "한련초"

            elif pred == "heonmyeongaju":
                print("prediction: 흰명아주")
                tmp = "흰명아주"

        if gt_label != tmp:
            cal = 0.0
        cal = float(cal)
        print_iou = "IoU=" + str(cal)

        print(cal)
        print(name)
        print("cal, name")
        #draw.text((int(bbox[0]), int(bbox[1]-270)), gt_label, (255,0,0), font=font)
        #draw.text((int(bboxes[0]), int(bboxes[1]-270)), tmp, (0,255,0), font=font)
        #draw.text((20, 0), print_iou, (255,0,0), font=fonts) #iou print
        #pil_image.save("./visualization/ex/%s.jpg"%name)
        mean += cal
        #final_result.append
        print(i)
        print("final iou:",float(mean)/38875)
        
    else:
        cal = 0.0
        cal = float(cal) 
        mean += cal
        #final_result.append
        print(i)
        print("final iou:",float(mean)/38875)   
