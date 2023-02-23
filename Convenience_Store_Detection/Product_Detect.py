from flask import Blueprint, request, send_file, jsonify
import time
import torch
import cv2
import pandas as pd
import pymysql

bp = Blueprint('', __name__, url_prefix='/product_detect')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model call
large_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/large_best.pt")
mini_beverage_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_beverage.pt")
mini_canned_food_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_canned_food.pt")
mini_coffee_tea_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_coffee_tea.pt")
mini_dessert_noodle_dairy_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_dessert_noodle_dairy.pt")
mini_drink_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_drink.pt")
mini_hmr_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_hmr.pt")
mini_snack_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_snack.pt")

# inference Settings
large_model.conf = 0.3
large_model.iou = 0.45
mini_beverage_model.conf = 0.3
mini_beverage_model.iou = 0.45
mini_canned_food_model.conf = 0.3
mini_canned_food_model.iou = 0.45
mini_coffee_tea_model.conf = 0.3
mini_coffee_tea_model.iou = 0.45
mini_dessert_noodle_dairy_model.conf = 0.3
mini_dessert_noodle_dairy_model.iou = 0.45
mini_drink_model.conf = 0.3
mini_drink_model.iou = 0.45
mini_hmr_model.conf = 0.3
mini_hmr_model.iou = 0.45
mini_snack_model.conf = 0.3
mini_snack_model.iou = 0.45

# use Nvidia cuda
large_model.to(device)
mini_beverage_model.to(device)
mini_canned_food_model.to(device)
mini_coffee_tea_model.to(device)
mini_dessert_noodle_dairy_model.to(device)
mini_drink_model.to(device)
mini_hmr_model.to(device)
mini_snack_model.to(device)

#데이터베이스 연결 전 데이터 기본 세팅
host_name = 'localhost'
port = 3306
user_name = 'root'
user_password = '1234'                     # <-------- 이거 비번 저랑 다르시면 수정 필요해요
database_name = 'product_detection'

detect_db = pymysql.connect( 
    host = host_name,
    port = port,
    user = user_name,
    passwd = user_password,
    db = database_name,
    charset = 'utf8'
)

cursor = detect_db.cursor()

def detect_inference(img_path):

    result_dict = {"result" : "true"}
    img = cv2.imread(img_path)

    detect = large_model(img, size=640)
    res_pd = detect.pandas().xyxy[0]

    """
        xmin      ymin        xmax        ymax  confidence  class        name
0  14.288334  3.687443  286.949371  287.652222    0.514725      7  coffee_tea
    
    형식으로 출력되며 empty를 통해서 비어있나 확인이 가능하다.
    """
    
    
    if res_pd.empty == False:
        
        detail_trigger = res_pd.loc[0, ['class']].item() #.item()으로 뽑아줘야 판다스가 아닌 기본자료형으로 뽑힌다.

        if detail_trigger == 0:
            
            print("0 번 라벨 들어옴")
            detail_detact = mini_snack_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]

            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                result_dict["result"] = "false"
                return result_dict

        elif detail_trigger == 1 or detail_trigger == 2 or detail_trigger == 4:
            
            detail_detact = mini_dessert_noodle_dairy_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]

            print("1,2,4 번 라벨 들어옴")
            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                result_dict["result"] = "false"
                return result_dict

        elif detail_trigger == 3:
            
            print("3 번 라벨 들어옴")
            detail_detact = mini_hmr_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]

            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                result_dict["result"] = "false"
                return result_dict

        elif detail_trigger == 5:
            
            print("5 번 라벨 들어옴")
            detail_detact = mini_beverage_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]

            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                print(result_pd.loc[0, ['class']].item())
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                result_dict["result"] = "false"
                return result_dict

        elif detail_trigger == 6:
            
            print("6 번 라벨 들어옴")
            detail_detact = mini_drink_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]

            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                result_dict["result"] = "false"
                return result_dict

        elif detail_trigger == 7:
            
            print("7 번 라벨 들어옴")
            detail_detact = mini_coffee_tea_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]
            print(result_pd)

            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                print("못 잡냐?")
                result_dict["result"] = "false"
                return result_dict
            
        elif detail_trigger == 8:
            
            print("8 번 라벨 들어옴")
            detail_detact = mini_canned_food_model(img, size=640)
            result_pd = detail_detact.pandas().xyxy[0]

            if result_pd.empty == False:
                # 데이터베이스 입력
                sql_image = "INSERT INTO image (img_name, label_no) VALUES (%s, %s)"
                cursor.execute(sql_image, (img_path, result_pd.loc[0, ['class']].item()))
                detect_db.commit()
                detect_db.close()

                # bbox 다시 치기
                minX = result_pd.loc[0, ['xmin']].item()
                minY = result_pd.loc[0, ['ymin']].item()
                maxX = result_pd.loc[0, ['xmax']].item()
                maxY = result_pd.loc[0, ['ymax']].item()
                cv2.rectangle(img, (int(minX), int(minY)), (int(maxX), int(maxY)), (0,255,0), 2)
                cv2.imwrite(img_path, img)

                return result_dict

            else:
                result_dict["result"] = "false"
                return result_dict

        return result_dict
    else:
        # 상품을 검출하지 못하였을 경우
        result_dict["result"] = "false"
        return result_dict




@bp.route("/", methods=['GET','POST'])
def detect():
    
    if request.method == "POST":
        f = request.files['product_img']
        tm = str(time.time()).split(".")
        path = "./images/" + tm[0] + tm[1]  + ".jpg"
        f.save(path)
        dict = detect_inference(path)
        if dict["result"] == "false":
            # 객체 탐지 실패
            return jsonify(dict)
        else:
            # 객체 탐지 성공
            return send_file(path, mimetype='image/png')

@bp.route("/text", methods=['GET','POST'])
def detect_text():
    # 여기서 json 파일 값을 전해준다.
    # DB 구현 영역
     data = {'file' : 'jsontest', 'text' : 'hi! jsontest!'}
     return jsonify(data)