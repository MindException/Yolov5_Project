from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import holidays
from tqdm import tqdm
import itertools
import os
import glob
import warnings

warnings.filterwarnings('ignore')


def train_prophet(csv_path='.\\combined_data2.csv', target_price='배추_가격(원/kg)'):
    # 경로 저장 변수
    target_price_csv = target_price.split('(')[0]

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 날짜 타입으로 변환
    df['일자'] = df['일자'].astype('str')
    df['일자'] = pd.to_datetime(df['일자'])
    # df.info()

    df = df.iloc[:, 1:]
    # df = df[df['일자'] < datetime.strptime('20230225', '%Y%m%d')]   # 마지막 관측값 까지 -> 2612
    df = df[df.index < 2612]
    df = df.fillna(0)
    cols0 = ['건고추_가격(원/kg)', '깻잎_가격(원/kg)',
             '당근_가격(원/kg)', '대파_가격(원/kg)',
             '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
             '배추_가격(원/kg)', '백다다기_가격(원/kg)',
             '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
             '시금치_가격(원/kg)', '애호박_가격(원/kg)',
             '양배추_가격(원/kg)', '양파_가격(원/kg)',
             '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
             '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
             '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']

    # target과 상관관계가 0.5 이상인 변수 추리기
    corr = df[cols0]
    corr = corr.corr(method='pearson')

    tp = corr[corr[f'{target_price}'] > 0.5].sort_values(f'{target_price}', ascending=False)
    tp = tp[f'{target_price}']
    x = list(tp.index)
    x.remove(f'{target_price}')

    df['weekday'] = df['일자'].dt.weekday
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    df['요일'] = df.apply(lambda x: weekday_list[x['weekday']], axis=1)
    # 0은 월요일 6은 일요일

    df = pd.concat([df, pd.get_dummies(df['요일'])], axis=1)
    df['ds'] = df['일자']
    df = df[['ds', '건고추_가격(원/kg)', '깻잎_가격(원/kg)',
             '당근_가격(원/kg)', '대파_가격(원/kg)',
             '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
             '배추_가격(원/kg)', '백다다기_가격(원/kg)',
             '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
             '시금치_가격(원/kg)', '애호박_가격(원/kg)',
             '양배추_가격(원/kg)', '양파_가격(원/kg)',
             '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
             '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
             '팽이버섯_가격(원/kg)', '포도_가격(원/kg)', '월', '화',
             '수', '목', '금', '토', '일']]
    feature = x
    framecols = ['ds', 'y']
    dayscol = ['월', '화', '수', '목', '금', '토', '일']
    dfcols = feature + dayscol + framecols

    # 휴일 객체
    kr_holidays = holidays.KR()

    # 휴일 컬럼 생성
    holiday_df = pd.DataFrame(columns=['ds', 'holiday'])
    holiday_df['ds'] = df['ds']
    holiday_df['holiday'] = holiday_df.ds.apply(lambda x: kr_holidays.get(x) if x in kr_holidays else 'non-holiday')
    # print(holiday_df)

    # df['holiday'] = df.ds.apply(lambda x: 1 if x in kr_holidays else 0)

    # 4주후의 가격을 target으로 설정
    df['y'] = df[target_price]
    df0 = df[dfcols]  # regressor 추가 모델용
    df['target'] = df[target_price].shift(-28)

    # grid Search Space
    # search_space = {
    #     'changepoint_prior_scale': [0.05, 0.1, 0.4, 0.5, 1.0, 5.0, 10.0],  # 추세의 유연성
    #     'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],  # 계절성의 유연성, 과적합 조절, 계절성의 영향
    #     'holidays_prior_scale': [0.05, 0.1, 1.0, 10.0],  # 휴일의 유연성, 영향
    #     'seasonality_mode': ['additive', 'multiplicative'],  # 증가 추세, additive는 폭에 변화 x, multiplicative는 폭에 변화
    #     'holidays': [holiday_df]
    # }
    # 최적 값
    search_space = {
        'changepoint_prior_scale': [0.05, 0.1],  # 추세의 유연성
        'seasonality_prior_scale': [10.0, 15.0],  # 계절성의 유연성, 과적합 조절, 계절성의 영향
        'holidays_prior_scale': [1.0, 3.0],  # 휴일의 유연성, 영향
        'seasonality_mode': ['additive'],  # 증가 추세, additive는 폭에 변화 x, multiplicative는 폭에 변화
        'holidays': [holiday_df]
    }
    # regressor add Space
    regressor_list = []
    for L in range(0, len(feature) + 1):
        for subset in itertools.combinations(feature, L):
            regressor_list.append(list(subset))

    param_combined = [dict(zip(search_space.keys(), v)) for v in itertools.product(*search_space.values())]

    for regressors in tqdm(regressor_list, colour='green'):
        mapes = []
        for param in param_combined:
            # print('params', param)
            _m = Prophet(**param)
            regressors2 = regressors + dayscol
            if regressors2 is not None:
                for regressor in regressors2:
                    _m.add_regressor(regressor)

            _m.fit(df0)
            _cv_df = cross_validation(_m, initial='1095 days', period='180 days', horizon='365 days',
                                      parallel='processes')
            _df_p = performance_metrics(_cv_df, rolling_window=1)
            mapes.append(_df_p['mape'].values[0])

        if os.path.exists(f'.\\temp_{target_price_csv}.csv'):
            temp = pd.DataFrame(param_combined)
            temp['mapes'] = mapes
            temp['regressors'] = ', '.join(regressors)
            tuning_results = pd.concat([tuning_results, temp], axis=0)
            tuning_results.to_csv(f'.\\params_prophet_{target_price_csv}.csv', encoding='utf-8-sig')
        else:
            tuning_results = pd.DataFrame(param_combined)
            tuning_results['mapes'] = mapes
            tuning_results['regressors'] = ', '.join(regressors)
            tuning_results.to_csv(f'.\\temp_{target_price_csv}.csv', encoding='utf-8-sig')


def test(params_csv='.\\prophet', csv_path='.\\combined_data2.csv'):
    # 한글 폰트
    from matplotlib import font_manager, rc

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
    df['ds'] = df['일자']
    df = df[['ds', '건고추_가격(원/kg)', '깻잎_가격(원/kg)',
             '당근_가격(원/kg)', '대파_가격(원/kg)',
             '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
             '배추_가격(원/kg)', '백다다기_가격(원/kg)',
             '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
             '시금치_가격(원/kg)', '애호박_가격(원/kg)',
             '양배추_가격(원/kg)', '양파_가격(원/kg)',
             '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
             '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
             '팽이버섯_가격(원/kg)', '포도_가격(원/kg)', '월', '화',
             '수', '목', '금', '토', '일']]

    # 휴일 객체
    kr_holidays = holidays.KR()

    # 휴일 컬럼 생성
    holiday_df = pd.DataFrame(columns=['ds', 'holiday'])
    holiday_df['ds'] = df['ds']
    holiday_df['holiday'] = holiday_df.ds.apply(lambda x: kr_holidays.get(x) if x in kr_holidays else 'non-holiday')

    # 파라미터 불러오기
    params_list = glob.glob(os.path.join(params_csv, '*.csv'))
    result = pd.DataFrame(columns=['species', 'result_mae'])

    for path in params_list:
        if 'params_prophet' in os.path.basename(path):
            species = os.path.basename(path).split('_')[2]

        else:
            species = os.path.basename(path).split('_')[1]

        target_price = species + '_가격(원/kg)'
        cols0 = ['건고추_가격(원/kg)', '깻잎_가격(원/kg)',
                 '당근_가격(원/kg)', '대파_가격(원/kg)',
                 '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
                 '배추_가격(원/kg)', '백다다기_가격(원/kg)',
                 '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
                 '시금치_가격(원/kg)', '애호박_가격(원/kg)',
                 '양배추_가격(원/kg)', '양파_가격(원/kg)',
                 '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
                 '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
                 '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']

        # target과 상관관계가 0.5 이상인 변수 추리기
        corr = df[cols0]
        corr = corr.corr(method='pearson')

        tp = corr[corr[f'{target_price}'] > 0.5].sort_values(f'{target_price}', ascending=False)
        tp = tp[f'{target_price}']
        x = list(tp.index)
        x.remove(f'{target_price}')

        feature = x
        framecols = ['ds', 'y']
        dayscol = ['월', '화', '수', '목', '금', '토', '일']

        # target, feature 설정
        df['y'] = df[target_price]
        df0 = df  # regressor 추가 모델용

        # test용
        test_y = df.iloc[-28:, :][[target_price]]
        test_X = df.iloc[-28:, :]

        # 파라미터 불러오기
        params_df = pd.read_csv(path, encoding='utf-8-sig')
        min_df = params_df[params_df['mapes'] == params_df.min(axis=0)['mapes']]
        params_dict = {
            'changepoint_prior_scale': min_df['changepoint_prior_scale'].values[0],
            'seasonality_prior_scale': min_df['seasonality_prior_scale'].values[0],
            'holidays_prior_scale': min_df['holidays_prior_scale'].values[0],
            'seasonality_mode': min_df['seasonality_mode'].values[0],
            'holidays': holiday_df
        }
        # print(params_dict)

        if 'params_prophet' in os.path.basename(path):
            try:
                regressors = min_df['regressors'].values[0].split(', ')
            except:
                if min_df['regressors'].values[0]:
                    print(min_df['regressors'].values[0])
                    print(path)
                    regressors = []
                else:
                    print(min_df['regressors'].values[0])
                    print('error')
                    print(path)
                    exit()


        # 학습
        _m = Prophet(**params_dict)
        if regressors:
            regressors2 = regressors + dayscol
            for regressor in regressors2:
                _m.add_regressor(regressor)
        else:
            for regressor in dayscol:
                _m.add_regressor(regressor)
        _m.fit(df0)

        # 테스트
        forecast = _m.predict(test_X)
        pred_y = forecast['yhat']

        MAE = mean_absolute_error(test_y, pred_y)

        insert_dict = {'species': species, 'result_mae': MAE}
        result.append(insert_dict, ignore_index=True)

        plt.figure(figsize=(20, 10), dpi=300)
        plt.title('Prophet 예측 결과' + ' Valid MAE : ' + str(MAE)[:7])
        plt.ylabel('가격')
        plt.plot(df[target_price], alpha=0.9, label='Real')
        plt.plot(_m.predict(df)['yhat'], alpha=0.6, linestyle="--", label='Predict')
        plt.legend()
        plt.savefig(f'.\\prophet_result\\{species}.png', dpi=80)
    result.to_csv('.\\prophet_result\\prophet.csv', encoding='utf-8-sig')



# 실험적 모델링
# df_prophet = Prophet(changepoint_prior_scale=0.4, yearly_seasonality=10)
# df_prophet.fit(df)
#
# df_forecast = df_prophet.make_future_dataframe(periods=28, freq='D')
# df_forecast = df_prophet.predict(df_forecast)
#
# df_predict = df2[-56:]
# test_y = df_predict[:28]['target']
# test_yhat = df_forecast[df_forecast.ds >= datetime.strptime('20230225', '%Y%m%d')]['yhat']
#
# df_prophet.plot(df_forecast, xlabel='Date', ylabel='Price')
# plt.show()
#
# df_prophet.plot_components(df_forecast)
# plt.show()

# 마지막 테스트 평가
# MAE = mean_absolute_error(test_y, test_yhat)
# plt.figure(figsize=(20, 10), dpi=300)
# plt.title('Prophet 예측 결과'+   '   MAE : ' + str(MAE)[:7])
# plt.ylabel('가격')
# plt.plot(np.array(df['y']), alpha = 0.9, label = 'Real')
# plt.plot(df_forecast[df_forecast.ds < datetime.strptime('20230225', '%Y%m%d')]['yhat'], alpha = 0.6, linestyle = "--", label = 'Predict')
# plt.legend()
# plt.show()

if __name__ == '__main__':
    price_list = ['당근_가격(원/kg)', '대파_가격(원/kg)',
                  '마늘_가격(원/kg)', '무_가격(원/kg)',
                  '시금치_가격(원/kg)', '애호박_가격(원/kg)',
                  '양배추_가격(원/kg)', '양파_가격(원/kg)',
                  '청상추_가격(원/kg)',
                  '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
                  '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']
    # for price in price_list:
    #     train_prophet(target_price=price)
    train_prophet(target_price='배추_가격(원/kg)')
    # test()
