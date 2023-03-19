import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
# 한글 폰트
from matplotlib import font_manager, rc


def xgb_gridsearchcv(csv_path='.\\combined_data2.csv', target_price='배추_가격(원/kg)'):
    font_path = 'C:/Windows/Fonts/batang.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
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
    # print(df)

    # 배추 가격 예측을 위해 배추 가격의 분해 시계열의 잔차를 feature로 이용
    df['resid'] = 0
    stl = STL(df[['일자', f'{target_price}']].set_index('일자'), period=12)
    res = stl.fit()
    df['resid'] = res.resid.values

    # feature, target 설정
    feature = df.columns[2:]

    # 예측할 작물의 가격의 4주치를 target에 설정
    df['target'] = df[f'{target_price}'].shift(-7)

    df_learn = df[:-35]
    df_predict = df[-35:]
    train_X = df_learn[feature]
    train_y = df_learn['target']
    test_X = df_predict[:-7][feature]
    test_y = df_predict[:-7]['target']

    # random forest grid search
    model = XGBRegressor()
    params = {'eval_metric': ['mae'],
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # so called `eta` value
              'max_depth': [5, 6, 9],
              'min_child_weight': [1, 3],
              'n_estimators': [100, 300, 500]}
    tscv = TimeSeriesSplit(n_splits=4)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=params, scoring='neg_mean_absolute_error', n_jobs=-1)
    gsearch.fit(train_X, train_y)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    best_params = gsearch.best_params_

    # 테스트 시각화
    y_pred = best_model.predict(test_X)
    R2 = best_model.score(test_X, test_y)
    MAE = mean_absolute_error(test_y, y_pred)
    print(MAE)
    df_params = pd.DataFrame([best_params])
    df_params['MAE'] = MAE
    csv_target = target_price.split('(')[0]
    df_params.to_csv(f'.\\xgb\\{csv_target}_xgb_bestparams.csv', encoding='utf-8-sig')

    # plt.figure(figsize=(20, 10), dpi=300)
    # plt.title('RandomForest 예측 결과' + '   MAE : ' + str(MAE)[:7])
    # plt.ylabel('가격')
    # plt.plot(np.array(train_y), alpha=0.9, label='Real')
    # plt.plot(best_model.predict(train_X), alpha=0.6, linestyle="--", label='Predict')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    price_list = ['당근_가격(원/kg)', '대파_가격(원/kg)',
                  '마늘_가격(원/kg)', '무_가격(원/kg)',
                  '시금치_가격(원/kg)', '애호박_가격(원/kg)', '배추_가격(원/kg)',
                  '양배추_가격(원/kg)', '양파_가격(원/kg)',
                  '청상추_가격(원/kg)',
                  '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
                  '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']
    for i in price_list:
        xgb_gridsearchcv(target_price=i)
