# coco(json파일)을 yolo형식으로 바꾼다.
#  yolo 형식은
# 파일명 -> 이미지와 동일 이름.txt
# 형식 예시) -> 2 0.534375 0.35833333333333334 0.0625 0.0875
#              라벨(숫자), 중앙x값, 중앙y값, 너비, 높이

import os
import glob
import json
import shutil
import cv2

def find_json_file(json_folder_path):
    all_root = []

    for (path, dir, files) in os.walk(json_folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]    #을 통하여 파일 확장자명 리스트를 찾는다.
            # ext -> json을 찾는다.
            if ext == ".json":
                root = os.path.join(path, filename)
                # json 파일의 경로가 생성되어진다.
                all_root.append(root)

    return all_root


folder_path = "./wine_labels_coco"
json_paths = find_json_file(folder_path)   # 각 위치의 json 파일을 불러온다.

for json_path in json_paths:
    
    #json 파일 열기
    with open(json_path, "r") as file:
        json_data = json.load(file)
        
    #json 파일 라벨 읽기
    labels = {}     #라벨 딕션어리 만들기
    
    for category in json_data["categories"]:
        labels[category["id"]] = category["name"]   #딕션어리에 라벨 추가
        
    # 아래와 같이 추가됨
    # {0: 'wine-labels', 1: 'AlcoholPercentage', 2: 'Appellation AOC DOC AVARegion', 3: 'Appellation QualityLevel', 4: 'CountryCountry', 
    # 5: 'Distinct Logo', 6: 'Established YearYear', 7: 'Maker-Name', 8: 'Organic', 9: 'Sustainable', 
    # 10: 'Sweetness-Brut-SecSweetness-Brut-Sec', 11: 'TypeWine Type', 12: 'VintageYear'}

    # json 파일 형태가 images에 파일 이름과 id 번호를 찾아서, annotations에서 bbox를 꺼내야 한다.
    # 5126 줄

    # 현재 폴더를 뽑는다.
    now_folder = json_path.split("\\")[1]

    # 현재 폴더의 파일이름들이다.
    file_list = []
    for (path, dir, files) in os.walk("./wine_labels_coco/" + now_folder):
        file_list = files

    images_info_list = json_data["images"]

    id_dict = {} # 번호 : 파일 이름
    for image_info in images_info_list:
        if image_info["file_name"] in file_list:
            id_dict[image_info["id"]] = image_info["file_name"]         #번호와 딕션어리 추가


    # 이제 bbox list가 들어있는 딕션어리 만들기
    bbox_dict = {}
    for id_num in list(id_dict.keys()):
        bbox_dict[id_num] = []      #bbox들을 담을 리스트를 생성


    # bbox 추가하기
    for dict in json_data["annotations"]:
        bbox_dict[dict["image_id"]].append(dict["category_id"])              # 라벨 추가
        bbox_dict[dict["image_id"]].append(dict["bbox"])            # bbox 추가

    # 옮길 directory
    new_dataset_path = "./new_wine_labels_coco"
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # train,val,test
    dataset_types_path = new_dataset_path + "/" + json_path.split("\\")[1]
    if not os.path.exists(dataset_types_path):
        os.makedirs(dataset_types_path)

    # yolo 형식으로 변환
    # labels 폴더 생성
    dataset_labels_path = dataset_types_path + "/labels"
    if not os.path.exists(dataset_labels_path):
        os.makedirs(dataset_labels_path)

    # 파일에 데이터 저장
    # 형식 -> 2 0.534375 0.35833333333333334 0.0625 0.0875
    # 딕션어리는 items로 돌아야 다 가져온다.
    for meta_data_key, meta_data_bbox in bbox_dict.items():

        file_name = id_dict[meta_data_key].split(".")[:-1]
        file_name = ".".join(file_name)

        with open(f"{dataset_labels_path}/{file_name}.txt", 'a') as f:
            # 홀수는 라벨 번호, 짝수는 bbox 정보가 들어잇다.
            for i, meta_data in enumerate(meta_data_bbox):
                if (i+1) % 2 == 1:
                    # 홀수일 경우 라벨 추가
                    f.write(f"{meta_data} ")

                else:
                    # 짝수일 경우 bbox 추가

                    # yolo 형식으로 파일 변환 0~1 사이로 정규화 시켜야 한다.
                    img_shape_path = "./wine_labels_coco" + "/" + json_path.split("\\")[1] + "/" + f"{file_name}.jpg"
                    img = cv2.imread(img_shape_path)

                    h, w, c = img.shape

                    # center는 최대 최소 더하고 나누기 2가 제일 정확
                    # json 파일과 비교 결과 최소값x, 최소값y, 변화량x, 변화량y
                    center_x = ((meta_data[0] + (meta_data[0] + meta_data[2])) / 2) / w
                    center_y = ((meta_data[1] + (meta_data[1] + meta_data[3])) / 2) / h
                    object_width = meta_data[2] / w
                    object_height = meta_data[3] / h

                    f.write(f"{center_x} {center_y} {object_width} {object_height} \n")

    print(f"{json_path}라벨 생성 완료")

    # 이미지 데이터 새 폴더에 옮기기
    # 이미지 폴더 생성
    dataset_images_path = dataset_types_path + "/images"
    if not os.path.exists(dataset_images_path):
        os.makedirs(dataset_images_path)

    # 이미지 경로 불러오기
    origin_path = "./wine_labels_coco" + "/" + json_path.split("\\")[1]
    img_files = glob.glob(os.path.join(origin_path, '*.jpg'))

    # 이미지 이동
    for img_path in img_files:
        # 나눠질 모양
        # ['./wine_labels_coco/test', '100140_jpg.rf.cb3528cde50580e4305be1a8c27af1b3.jpg']

        img_name = img_path.split("\\")[1]

        # 이동 전 경로
        origin_img_path = img_path.split("\\")[0] + "/" + img_name
        # 이동 후 경로
        move_img_path = dataset_images_path + "/" + img_name

        # 파일 이동
        shutil.move(origin_img_path, move_img_path)

    print(f"{json_path} 이미지 이동 완료")