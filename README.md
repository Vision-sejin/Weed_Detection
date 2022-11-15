Hello it is for Weed Detection with MMDet

I train the weed with yolof and the result is ./epoch_12.pth("https://drive.google.com/file/d/1Kl4gvwLpAhN53-wmJOL1h4WtDng9DlAt/view?usp=sharing)

I use weed data from "https://www.aihub.or.kr/" (weed data)

And our base code(but changed) is from "https://github.com/open-mmlab/mmdetection"

As you see, the result example is 'result_example.jpg'

![result_example](https://user-images.githubusercontent.com/117714660/200505740-64291ff4-81db-44a1-abba-0ec926fe3925.jpg)

And if you want to visualize the prediction and gt bbox and name, then you can use visualization.py 

or if you want to check only iou, then you can just use notimage.py
#you need little change another code 



#Train
python tools/train.py configs/yolox/yolox_s_8x8_300e_coco.py --gpu-id 1




#Test
python tools/test.py configs/yolof/yolof_r50_c5_8x8_1x_coco.py work_dirs/yolof_r50_c5_8x8_1x_coco/epoch_12.pth --show-dir ../../../8T/yolof_result



#Visualization
python visualization.py # in ./visualization/*

