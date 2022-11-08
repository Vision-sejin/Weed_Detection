from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import json
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np


# Specify the path to model config and checkpoint file
config_file = 'configs/yolof/yolof_r50_c5_8x8_1x_coco.py'
checkpoint_file = './work_dirs/yolof_r50_c5_8x8_1x_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


source = "../ai_plant_data/traindata/data/Validation/"
img_dir= "../val"



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
    img = "../val/"+name + ".jpg"
    label = source + choice_label + name+ ".json"

    with open(label, 'r') as f:
        json_data = json.load(f)

    bbox = json_data['annotations']['bbox'] #['annotations']['bbox']
    gt_label = json_data['annotations']['weeds_kind']  #['categories]['name']
    
    if gt_label == "ganki" :
        gt_label == "강피"


    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))

    print(pt1)
    print(gt_label)

    result = inference_detector(model, img)
    # visualize the results in a new window
    #img = model.show_result(img, result, bbox_color=[(0,255,0),(0,0,0)], text_color=(255, 255, 255), thickness=10, font_size=42)
    #mmcv.show(img)
    model.show_result(img, result, bbox_color=[(0,255,0),(0,0,0)], text_color=(255, 255, 255), out_file='./all/%s.jpg'%name, thickness=10, font_size=42)

    image = cv2.imread("./all/%s.jpg"%name)
    image = cv2.rectangle(image, pt1, pt2, (0,0,255), 10)

    pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(pil_image)
    fontpath = "../../usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"
    font = ImageFont.truetype(fontpath, 100)
    draw = ImageDraw.Draw(pil_image)


    draw.text((int(bbox[0]), int(bbox[1]-110)), str(gt_label), (255,0,0), font=font)
    pil_image.save("./visualization/%s.jpg"%name)










    #image = cv2.putText(image, "%s"%gt_label, (int(bbox[0]), int(bbox[1]-30)), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 4, cv2.LINE_8)



    #image = cv2.putText(image, "green = prediction", (0, 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    #image = cv2.putText(image, "red = ground truth", (0 ,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

    #print(image.shape)
    cv2.imwrite("./all/%s.jpg"%name, image)

    #print(root)



# test a single image and show the results
#name = "20210819_RGB_A_011_45_31"
#root = name[13]
#choice_label = "label/VL1/ganpi/"





#print(img)

#print(label)




#cv2.imshow(img)


# or save the visualization results to image files
#model.show_result(img, result, out_file='./demo_result.jpg')
