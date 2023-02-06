import os
import glob
import cv2
from tqdm import tqdm

# D:\dataset\Training\labels\40029\ 40029_00_m_1.jpg

def resizing_img_996(path):
    file_paths = glob.glob(os.path.join(path, 'images', "*", '*.jpg'))

    for img_path in tqdm(file_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (996, 996))
        cv2.imwrite(img_path, img)

if __name__=="__main__":
    cur_path = "D:\dataset"
    resizing_img_996(cur_path + "\\Training")
    resizing_img_996(cur_path + "\\Validation")
