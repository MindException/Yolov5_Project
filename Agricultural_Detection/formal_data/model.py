import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
import joblib
import warnings

warnings.filterwarnings('ignore')

# 모델 성능 평가
def compare_model_mae():
    rf_path = glob.glob(os.path.join('.\\rf', '*.csv'))
    xgb_path = glob.glob(os.path.join('.\\xgb', '*.csv'))

    rf, xgb = [], []
    for path in rf_path:
        df = pd.read_csv(path, encoding='utf-8-sig')
        mae = df.iloc[0, :]['MAE']
        rf.append(mae)
    for path in xgb_path:
        df = pd.read_csv(path, encoding='utf-8-sig')
        mae = df.iloc[0, :]['MAE']
        xgb.append(mae)
    print('Random Forest의 일주일 뒤 평균 농작물 가격 MAE는 ', np.mean(rf))
    print('XGBoost의 일주일 뒤 평균 농작물 가격 MAE는 ', np.mean(xgb))


def train_save_model(target_price='배추_가격(원/kg)', save_dir='.\\final_model', plot_tf=False, modelz='both'):
    # modelz : rf, xgb, both
    # 한글 폰트
    from matplotlib import font_manager, rc

    font_path = 'C:/Windows/Fonts/NGULIM.TTF'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    target_type = target_price.split('_')[0]
    df = pd.read_csv('.\\combined_data2.csv', encoding='utf-8-sig')
    # 날짜 타입으로 변환
    df['일자'] = df['일자'].astype('str')
    df['일자'] = pd.to_datetime(df['일자'])
    # df.info()

    df = df.iloc[:, 1:]
    # df = df[df['일자'] < datetime.strptime('20230225', '%Y%m%d')]   # 마지막 관측값 까지 -> 2612
    df = df[df.index < 2612]
    df = df.fillna(0)

    df['weekday'] = df['일자'].dt.weekday
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    df['요일'] = df.apply(lambda x: weekday_list[x['weekday']], axis=1)
    # 0은 월요일 6은 일요일

    df = pd.concat([df, pd.get_dummies(df['요일'])], axis=1)
    df = df[['일자', '요일', '건고추_거래량(kg)', '건고추_가격(원/kg)', '깻잎_거래량(kg)', '깻잎_가격(원/kg)',
             '당근_거래량(kg)', '당근_가격(원/kg)', '대파_거래량(kg)', '대파_가격(원/kg)', '마늘_거래량(kg)',
             '마늘_가격(원/kg)', '무_거래량(kg)', '무_가격(원/kg)', '미나리_거래량(kg)', '미나리_가격(원/kg)',
             '배추_거래량(kg)', '배추_가격(원/kg)', '백다다기_거래량(kg)', '백다다기_가격(원/kg)',
             '새송이_거래량(kg)', '새송이_가격(원/kg)', '샤인머스캇_거래량(kg)', '샤인머스캇_가격(원/kg)',
             '시금치_거래량(kg)', '시금치_가격(원/kg)', '애호박_거래량(kg)', '애호박_가격(원/kg)',
             '양배추_거래량(kg)', '양배추_가격(원/kg)', '양파_거래량(kg)', '양파_가격(원/kg)',
             '얼갈이배추_거래량(kg)', '얼갈이배추_가격(원/kg)', '청상추_거래량(kg)', '청상추_가격(원/kg)',
             '토마토_거래량(kg)', '토마토_가격(원/kg)', '파프리카_거래량(kg)', '파프리카_가격(원/kg)',
             '팽이버섯_거래량(kg)', '팽이버섯_가격(원/kg)', '포도_거래량(kg)', '포도_가격(원/kg)', '월', '화',
             '수', '목', '금', '토', '일']]

    # 배추 가격 예측을 위해 배추 가격의 분해 시계열의 잔차를 feature로 이용
    df['resid'] = 0
    stl = STL(df[['일자', f'{target_price}']].set_index('일자'), period=12)
    res = stl.fit()
    df['resid'] = res.resid.values

    # feature, target 설정
    feature = df.columns[2:]

    # 예측할 작물의 1일 뒤 가격을 target에 설정
    df['target'] = df[f'{target_price}'].shift(-1)

    # 최근 4주의 데이터를 validation dataset으로 활용
    df_learn = df[:-29]
    df_predict = df[-29:]
    train_X = df_learn[feature]
    train_y = df_learn['target']
    test_X = df_predict[:-1][feature]
    test_y = df_predict[:-1]['target']

    # 모델 (랜덤포레스트, XGBoost 사용)
    # 파라미터 소환
    xgb_params = os.path.join('.\\xgb', f'{target_type}_가격_xgb_bestparams.csv')
    rf_params = os.path.join('.\\rf', f'{target_type}_가격_rf_bestparams.csv')
    xgb_df = pd.read_csv(xgb_params, encoding='utf-8-sig')
    xgb_df = xgb_df.iloc[:, 1:]
    rf_df = pd.read_csv(rf_params, encoding='utf-8-sig')
    rf_df = rf_df.iloc[:, 1:]

    params_xgb = list(xgb_df.columns)
    params_xgb.remove('MAE')
    values_xgb = [xgb_df[key][0] for key in params_xgb]
    xgb_dict = {k: v for k, v in zip(params_xgb, values_xgb)}

    params_rf = list(rf_df.columns)
    params_rf.remove('MAE')
    values_rf0 = [rf_df[key][0] for key in params_rf]
    values_rf = []
    for i in values_rf0:
        if pd.isna(i):
            i = None
        values_rf.append(i)
    rf_dict = {k: v for k, v in zip(params_rf, values_rf)}
    if modelz == 'rf':
        model1 = RandomForestRegressor(**rf_dict)
        model1.fit(train_X, train_y)

        y_pred = model1.predict(test_X)
        MAE1 = mean_absolute_error(test_y, y_pred)

        result = pd.DataFrame(columns=['True_price', 'rf_predicted_price'])
        result['True_price'] = df['target']
        result['rf_predicted_price'] = model1.predict(df[feature])
        result.to_csv(os.path.join(save_dir, f'rf_{target_type}.csv'), encoding='utf-8-sig')
        joblib.dump(model1, os.path.join(save_dir, 'models', 'rf', f'{target_type}_rf_model.pkl'))

        if plot_tf:
            plt.figure(figsize=(20, 10), dpi=300)
            plt.title('RandomForest 예측 결과' + ' Valid MAE : ' + str(MAE1)[:7])
            plt.ylabel('가격')
            plt.plot(np.array(df['target']), alpha=0.9, label='Real')
            plt.plot(model1.predict(df[feature]), alpha=0.6, linestyle="--", label='Predict')
            plt.legend()
            plt.show()


    # XGBoost
    elif modelz == 'xgb':
        model2 = XGBRegressor(**xgb_dict)
        model2.fit(train_X, train_y)

        y_pred = model2.predict(test_X)
        MAE2 = mean_absolute_error(test_y, y_pred)

        result = pd.DataFrame(columns=['True_price', 'xgb_predicted_price'])
        result['True_price'] = df['target']
        result['xgb_predicted_price'] = model2.predict(df[feature])
        result.to_csv(os.path.join(save_dir, f'xgb_{target_type}.csv'), encoding='utf-8-sig')
        model2.save_model(os.path.join(save_dir, 'models', 'xgb', f'{target_type}_xgb_model.bst'))

        if plot_tf:
            plt.figure(figsize=(20, 10), dpi=300)
            plt.title('XGBoost 예측 결과' + ' Valid MAE : ' + str(MAE2)[:7])
            plt.ylabel('가격')
            plt.plot(np.array(df['target']), alpha=0.9, label='Real')
            plt.plot(model2.predict(df[feature]), alpha=0.6, linestyle="--", label='Predict')
            plt.legend()
            plt.show()


    elif modelz == 'both':
        model1 = RandomForestRegressor(**rf_dict)
        model1.fit(train_X, train_y)

        y_pred = model1.predict(test_X)
        MAE1 = mean_absolute_error(test_y, y_pred)

        model2 = XGBRegressor(**xgb_dict)
        model2.fit(train_X, train_y)

        y_pred = model2.predict(test_X)
        MAE2 = mean_absolute_error(test_y, y_pred)

        result = pd.DataFrame(columns=['True_price', 'rf_predicted_price', 'xgb_predicted_price'])
        result['True_price'] = df['target']
        result['rf_predicted_price'] = model1.predict(df[feature])
        result['xgb_predicted_price'] = model2.predict(df[feature])
        result.to_csv(os.path.join(save_dir, f'rf_xgb_{target_type}.csv'), encoding='utf-8-sig')
        joblib.dump(model1, os.path.join(save_dir, 'models', 'rf', f'{target_type}_rf_model.pkl'))
        model2.save_model(os.path.join(save_dir, 'models', 'xgb', f'{target_type}_xgb_model.bst'))

        if plot_tf:
            plt.figure(figsize=(20, 10), dpi=300)
            plt.title('RandomForest 예측 결과' + ' Valid MAE : ' + str(MAE1)[:7])
            plt.ylabel('가격')
            plt.plot(np.array(df['target']), alpha=0.9, label='Real')
            plt.plot(model1.predict(df[feature]), alpha=0.6, linestyle="--", label='Predict')
            plt.legend()
            # plt.show()
            plt.savefig(f'{save_dir}\\graphs\\{target_type}_rf_graph.png', dpi=200)

            plt.figure(figsize=(20, 10), dpi=300)
            plt.title('XGBoost 예측 결과' + ' Valid MAE : ' + str(MAE2)[:7])
            plt.ylabel('가격')
            plt.plot(np.array(df['target']), alpha=0.9, label='Real')
            plt.plot(model2.predict(df[feature]), alpha=0.6, linestyle="--", label='Predict')
            plt.legend()
            # plt.show()
            plt.savefig(f'{save_dir}\\graphs\\{target_type}_xgb_graph.png', dpi=200)


if __name__ == '__main__':
    price_list = ['당근_가격(원/kg)', '대파_가격(원/kg)',
                  '마늘_가격(원/kg)', '무_가격(원/kg)',
                  '시금치_가격(원/kg)', '애호박_가격(원/kg)',
                  '양배추_가격(원/kg)', '양파_가격(원/kg)',
                  '청상추_가격(원/kg)', '배추_가격(원/kg)',
                  '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
                  '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']
    for i in tqdm(price_list, colour='green'):
        train_save_model(target_price=i, plot_tf=True)

