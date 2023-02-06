import os
import glob
import csv
import pandas as pd
from tqdm import tqdm


"""
폴더에 이름을 먼저 변경한다
전)  D:\dataset\Training\[라벨]음료2\40029_빅토리아라임350ML\40029_00_m_1.jpg
후)  D:\dataset\Training\labels\40029\ 40029_00_m_1.jpg

1. images와 label로 변경
2. 바뀐 이름 전부 저장
3. 폴더명 img_no만 남기기
4. 이미지 996 * 996으로 resize
"""

def chage_images_and_labels(path):
    for i, (path, dir, files) in enumerate(os.walk(path)):
        if i == 0:
            for dir_name in dir:
                if "라벨" in dir_name:
                    os.rename(path + "\\" + dir_name, path + "\\" + "labels")
                elif "원천" in dir_name:
                    os.rename(path + "\\" + dir_name, path + "\\" + "images")

def write_folder_name(path):
    img_no_path = path + "\\" +"Training" + "\\" + "images"
    for i, (path, dir, files) in enumerate(os.walk(img_no_path)):
        if i == 0:

            img_no = []
            img_name = []
            for dir_name in dir:
                split_name = dir_name.split("_")
                img_no.append(split_name[0])
                img_name.append(split_name[1])

            df_name = pd.DataFrame({"labelcode": img_no, "name": img_name})
            df_name.to_csv("음료목록라벨.csv", encoding='utf-8-sig') # sig로 하면 ms 액셀에서 안 깨짐


def rename_folder_to_img_no(path):

    # 이미지를 먼저 바꾼다.
    img_no_path = path + "\\" + "images"

    for i, (path, dir, files) in enumerate(os.walk(img_no_path)):
        if i == 0:

            for dir_name in tqdm(dir):
                split_name = dir_name.split("_")[0]
                origin_path = img_no_path + "\\" + dir_name
                new_path = img_no_path + "\\" + split_name
                os.rename(origin_path, new_path)

    # 라벨을 바꾼다.
    img_no_path2 = path + "\\" + "labels"

    for i, (path, dir2, files) in enumerate(os.walk(img_no_path2)):
        if i == 0:

            for dir_name2 in tqdm(dir2):
                split_name2 = dir_name2.split("_")[0]
                origin_path2 = img_no_path2 + "\\" + dir_name2
                new_path2 = img_no_path2 + "\\" + split_name2
                os.rename(origin_path2, new_path2)



if __name__=="__main__":
    cur_path = "D:\dataset"

    # 먼저 폴더 이름을 읽는다.
    write_folder_name(cur_path)

    # train
    train_path = cur_path + "\Training"
    chage_images_and_labels(train_path)
    rename_folder_to_img_no(train_path)

    # val
    val_path = cur_path + "\Validation"
    chage_images_and_labels(val_path)
    rename_folder_to_img_no(val_path)