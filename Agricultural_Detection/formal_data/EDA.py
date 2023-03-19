import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import math
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# 한글 폰트
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/NGULIM.TTF'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

df = pd.read_csv('.\\combined_data2.csv', encoding='utf-8-sig')
df = df.iloc[:, 1:]
df['일자'] = df['일자'].astype('str')
df['일자'] = pd.to_datetime(df['일자'])
# df.info()
df = df[df['일자'] < datetime.strptime('20230225', '%Y%m%d')]
df = df.fillna(0)
# print(df)
cols = ['일자', '건고추_거래량(kg)', '건고추_가격(원/kg)', '깻잎_거래량(kg)', '깻잎_가격(원/kg)',
        '당근_거래량(kg)', '당근_가격(원/kg)', '대파_거래량(kg)', '대파_가격(원/kg)', '마늘_거래량(kg)',
        '마늘_가격(원/kg)', '무_거래량(kg)', '무_가격(원/kg)', '미나리_거래량(kg)', '미나리_가격(원/kg)',
        '배추_거래량(kg)', '배추_가격(원/kg)', '백다다기_거래량(kg)', '백다다기_가격(원/kg)',
        '새송이_거래량(kg)', '새송이_가격(원/kg)', '샤인머스캇_거래량(kg)', '샤인머스캇_가격(원/kg)',
        '시금치_거래량(kg)', '시금치_가격(원/kg)', '애호박_거래량(kg)', '애호박_가격(원/kg)',
        '양배추_거래량(kg)', '양배추_가격(원/kg)', '양파_거래량(kg)', '양파_가격(원/kg)',
        '얼갈이배추_거래량(kg)', '얼갈이배추_가격(원/kg)', '청상추_거래량(kg)', '청상추_가격(원/kg)',
        '토마토_거래량(kg)', '토마토_가격(원/kg)', '파프리카_거래량(kg)', '파프리카_가격(원/kg)',
        '팽이버섯_거래량(kg)', '팽이버섯_가격(원/kg)', '포도_거래량(kg)', '포도_가격(원/kg)']
purchases = []
prices = []
for i, col in enumerate(cols[1:]):
    if i % 2 == 0:
        purchases.append(col)
    else:
        prices.append(col)

# # 시계열에 따른 가격 분포
# for i in prices[1:12]:
#     plt.plot(df['일자'], df[i], label=i)
# plt.ylim(0, 40500)
# plt.legend()
# plt.grid()
# plt.show()
#
# # 품목별 농산물 가격 분포
# colors = ['tab:blue', 'tab:brown', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:cyan']
# bx = plt.boxplot([df[i] for i in prices[1:9]], labels=prices[1:9], patch_artist=True)
# for i, color in enumerate(colors):
#     bx['boxes'][i].set_facecolor(color)
#     bx['boxes'][i].set_color(color)
# plt.ylim(0, 18500)
# plt.grid(color='grey', linestyle='-', linewidth=0.5, axis='y')
# plt.show()
#
# # 건고추 시계열 가격 분포
# plt.plot(df['일자'], df[prices[0]], label=prices[0])
# plt.grid()
# plt.show()

# # 배추의 가격과 거래량 곡선
# plt.plot(df['일자'], df['배추_가격(원/kg)'], label='배추_가격(원/kg)')
# plt.grid()
# plt.show()
# plt.plot(df['일자'], df['배추_거래량(kg)'], label='배추_거래량(kg)')
# plt.grid()
# plt.show()

# # 가격과 거래량의 상관관계
# corr = df[['건고추_거래량(kg)', '건고추_가격(원/kg)']]
# corr = corr.corr(method='pearson')
# sns.heatmap(corr)
# plt.show()
# print(corr)

corr = df[['건고추_가격(원/kg)', '깻잎_가격(원/kg)',
         '당근_가격(원/kg)', '대파_가격(원/kg)',
         '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
         '배추_가격(원/kg)', '백다다기_가격(원/kg)',
         '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
         '시금치_가격(원/kg)', '애호박_가격(원/kg)',
         '양배추_가격(원/kg)', '양파_가격(원/kg)',
         '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
         '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
         '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']]
corr = corr.corr(method='pearson')
# sns.heatmap(corr)
# plt.show()
tp = corr[corr['배추_가격(원/kg)'] > 0.5].sort_values('배추_가격(원/kg)', ascending=False)
print(tp['배추_가격(원/kg)'][:4])
x = list(tp.index)
x.remove('배추_가격(원/kg)')
print(x)

# 시계열 분해
# df['resid'] = 0
# stl = STL(df[['일자', '배추_가격(원/kg)']].set_index('일자'), period=12)
# res = stl.fit()
# df['resid'] = res.resid.values
#
# res.plot()
# plt.show()