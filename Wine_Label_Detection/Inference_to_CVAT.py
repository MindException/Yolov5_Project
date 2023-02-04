import torch
import os
import glob
import cv2
import xml.etree.ElementTree as ET
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

tree = ET.ElementTree()
root = ET.Element("annotations")
"""
<annotations>
</annotations>
"""
seen_count = 0

for img_path in image_path :
    # Image
    img = cv2.imread(img_path)

    # Inference
    results = model(img, size=640)

    # Results
    bbox = results.xyxy[0]
    # [368.72302, 345.64069, 465.59674, 472.57922, 0.69237, 4.00000] 이런 식으로 들어가 있는다.

    # image name
    image_name = os.path.basename(img_path) # 파일 이름이 아래와 같이 추출된다.
    # aditganteng_mp4-108_jpg.rf.3e18137f20fca45f61ddb86e12990e54.jpg

    # image w h
    h, w, c = img.shape

    # xml fix code
    # <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c479e.jpg" width="640" height="480">
    xml_frame = ET.SubElement(root, "image" , id="%d"%seen_count, name=image_name,
                              width="%d"%w, height="%d"%h)
    """
    <annotations>
        <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c479e.jpg" width="640" height="480">
        </image>
    </annotations>
    """

    for box in bbox :
        # <box label="car" occluded="0" source="manual" xtl="346.68"
        # ytl="325.63" xbr="427.46" ybr="404.08" z_order="0"> </box>
        # box
        """
        <annotations>
            <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c479e.jpg" width="640" height="480">
                <box></box>
                <box></box>
                <box></box>
                <box></box>
            </image>
            <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c43249e.jpg" width="640" height="480">
                <box></box>
                <box></box>
                <box></box>
                <box></box>
            </image>
        </annotations>
        """
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()

        # 소수점 3째 자리에서 반올림
        xtl = str(round(x1,3))
        ytl = str(round(y1,3))
        xbr = str(round(x2,3))
        ybr = str(round(y2,3))

        # clss
        clss_number = box[5].item()
        clss_number_int = int(clss_number)
        labels = label_dict[clss_number_int]        # 번호에 맞게 다시 라벨 변환

        # sc number
        sc = box[4].item()

        # bbox xml
        #  xtl="346.68" ytl="325.63" xbr="427.46" ybr="404.08" z_order="0"
        ET.SubElement(xml_frame, "box", label=labels, occluded="0", source="manual",
                      xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, z_order="0")

    seen_count +=1

tree._setroot(root)
tree.write("test_aug.xml", encoding="utf-8")