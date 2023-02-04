import torch
import os
import glob
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model call
model = torch.hub.load('ultralytics/yolov5', 'custom', path="./result_wine_model/weights/best.pt")

# inference Settings
model.conf = 0.5 # NMS confidence threshold
model.iou = 0.45 # NMS IoU threshold
model.to(device)

# image loader
image_dir = "./new_wine_labels_coco/test/images/"
image_path = glob.glob(os.path.join(image_dir, "*.jpg"))
label_dict ={0: 'wine-labels', 1: 'AlcoholPercentage', 2: 'Appellation AOC DOC AVARegion', 3: 'Appellation QualityLevel',
             4: 'CountryCountry', 5: 'Distinct Logo', 6: 'Established YearYear', 7: 'Maker-Name', 8: 'Organic', 9: 'Sustainable',
             10: 'Sweetness-Brut-SecSweetness-Brut-Sec', 11: 'TypeWine Type', 12: 'VintageYear'}


for img_path in image_path:
    # Image
    img = cv2.imread(img_path)

    # Inference
    results = model(img, size=640)

    # Results
    bbox = results.xyxy[0]
    # [368.72302, 345.64069, 465.59674, 472.57922, 0.69237, 4.00000] 이런 식으로 들어가 있는다.

    for box in bbox:
        minX = box[0].item()
        minY = box[1].item()
        maxX = box[2].item()
        maxY = box[3].item()

        cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
        cv2.imshow("img_bbox", img)

        while True:
            key = cv2.waitKey()
            if key == ord('s'):
                # 문제가 있는 사진 체크하기
                pass
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                exit()
            else:  # 위 if문에서 지정하지 않은 키보드 입력인 경우 다음 이미지로 넘어감
                break