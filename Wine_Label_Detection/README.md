# 와인 라벨 검출 프로젝트

본 프로젝트는 Yolov5를 이용하여 와인 라벨을 검출하는 프로젝트입니다.  
Yolov5를 사용하기 위한 제일 기본적인 프로젝트이며, 자세한 사용방법은 [YOLOv5 필기](https://github.com/MindException/MS_AI_School_WorkSpace/tree/main/YOLO) 에 존재하는 md파일을 통해 확인 가능합니다.

## 학습 목표

Yolov5를 사용하여 와인 라벨에 적힌 정보를 검출하는 제일 기본적인 프로젝트를 진행한다.  

## 코드 진행 순서

1. data폴더에 들어있는 이미지와 COCO 파일(json)들을 읽고 yolo에서 학습할 수 있는 dataset으로 변환한다. (__coco_to_Yolov5.py__ 를 사용)
2. [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 에서 yolov5 모델을 git clone하여 모델을 생성한다.
3. train.py에 def parse_opt를 수정한다.
4. train.py를 실행하여 모델을 학습한다.  
5. 추론(Inference)  
    * __Inference_eval.py__ 를 사용하여 학습된 모델에 test 데이터 값을 넣어 결과 bbox를 확인한다.  
    * __Inference_to_CVAT.py__ 를 사용하여 test의 bbox 결과값을 CVAT형식(__test_aug.xml__)으로 만들어서 Autolabel로 더 정확히 분석한다.