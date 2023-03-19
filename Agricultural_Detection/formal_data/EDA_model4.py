import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
import numpy as np
import holidays

# 한글 폰트
from matplotlib import font_manager, rc

font_path = 'C:/Windows/Fonts/NGULIM.TTF'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

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
# print(df)

# 배추 가격 예측을 위해 배추 가격의 분해 시계열의 잔차를 feature로 이용
df['resid'] = 0
stl = STL(df[['일자', '배추_가격(원/kg)']].set_index('일자'), period=12)
res = stl.fit()
df['resid'] = res.resid.values

# 휴일 객체
kr_holidays = holidays.KR()

# 휴일 컬럼 생성
df['holiday0'] = df['일자'].apply(lambda x: 'holiday' if x in kr_holidays else 'non-holiday')
df = pd.concat([df, pd.get_dummies(df['holiday0'])], axis=1)
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
         '수', '목', '금', '토', '일', 'resid', 'holiday', 'non-holiday']]

# feature, target 설정
feature = df.columns[2:]
# feature = ['배추_거래량(kg)', '배추_가격(원/kg)', '월', '화', '수', '목', '금', '토', '일']   # 성능이 더 낮음


# 예측할 작물의 1주일 뒤 가격을 target에 설정
df['target'] = df['배추_가격(원/kg)'].shift(-2)

# print(len(df)) 2612
# 최근 4주의 데이터를 validation dataset으로 활용
df_learn = df[:-30]
df_predict = df[-30:]
train_X = df_learn[feature]
train_y = df_learn['target']
test_X = df_predict[:-2][feature]
test_y = df_predict[:-2]['target']

# 네 번째 탐색적 모델 (랜덤포레스트, XGBoost 사용)
model = RandomForestRegressor()
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
# R2 = model.score(X_train, y_train)
MAE = mean_absolute_error(test_y, y_pred)

plt.figure(figsize=(20, 10), dpi=300)
plt.title('RandomForest 예측 결과' + ' Valid MAE : ' + str(MAE)[:7])
plt.ylabel('가격')
plt.plot(np.array(df['target']), alpha=0.9, label='Real')
plt.plot(model.predict(df[feature]), alpha=0.6, linestyle="--", label='Predict')
plt.legend()
plt.show()

# XGBoost
model = XGBRegressor()
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
# R2 = model.score(X_train, y_train)
MAE = mean_absolute_error(test_y, y_pred)

plt.figure(figsize=(20, 10), dpi=300)
plt.title('XGBoost 예측 결과' + ' Valid MAE : ' + str(MAE)[:7])
plt.ylabel('가격')
plt.plot(np.array(df['target']), alpha=0.9, label='Real')
plt.plot(model.predict(df[feature]), alpha=0.6, linestyle="--", label='Predict')
plt.legend()
plt.show()
