import glob
import sys
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

# 한글 폰트
from matplotlib import font_manager, rc

font_path = 'C:/Windows/Fonts/gulim.ttc'
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
df = df.replace(0, np.NaN)
cols0 = ['건고추_거래량(kg)', '건고추_가격(원/kg)', '깻잎_거래량(kg)', '깻잎_가격(원/kg)',
         '당근_거래량(kg)', '당근_가격(원/kg)', '대파_거래량(kg)', '대파_가격(원/kg)', '마늘_거래량(kg)',
         '마늘_가격(원/kg)', '무_거래량(kg)', '무_가격(원/kg)', '미나리_거래량(kg)', '미나리_가격(원/kg)',
         '배추_거래량(kg)', '배추_가격(원/kg)', '백다다기_거래량(kg)', '백다다기_가격(원/kg)',
         '새송이_거래량(kg)', '새송이_가격(원/kg)', '샤인머스캇_거래량(kg)', '샤인머스캇_가격(원/kg)',
         '시금치_거래량(kg)', '시금치_가격(원/kg)', '애호박_거래량(kg)', '애호박_가격(원/kg)',
         '양배추_거래량(kg)', '양배추_가격(원/kg)', '양파_거래량(kg)', '양파_가격(원/kg)',
         '얼갈이배추_거래량(kg)', '얼갈이배추_가격(원/kg)', '청상추_거래량(kg)', '청상추_가격(원/kg)',
         '토마토_거래량(kg)', '토마토_가격(원/kg)', '파프리카_거래량(kg)', '파프리카_가격(원/kg)',
         '팽이버섯_거래량(kg)', '팽이버섯_가격(원/kg)', '포도_거래량(kg)', '포도_가격(원/kg)']
for col in cols0:
    df[col] = df[col].interpolate(method='linear').fillna(0)

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

# feature, target 설정
feature = ['배추_거래량(kg)', '배추_가격(원/kg)', '월', '화', '수', '목', '금', '토', '일', 'resid']

# 예측할 작물의 가격의 4주치를 target에 설정
df['target'] = df['배추_가격(원/kg)'].shift(-29)

scaler = MinMaxScaler()
df[feature] = scaler.fit_transform(df[feature])

df_learn = df[:-58]
# for plot
df_learn_plot_X = df_learn[feature]
plot_tensor = torch.FloatTensor(df_learn_plot_X.values)

df_predict = df[-58:]
df_train = df_learn[:round(-0.1 * len(df_learn))]
df_valid = df_learn[round(-0.1 * len(df_learn)):]
train_X = df_train[feature]
train_y = df_train['target']
valid_X = df_valid[feature]
valid_y = df_valid['target']
test_X = df_predict[:29][feature]
test_y = df_predict[:29]['target']


# 네 번째 탐색적 모델 (LSTM)
trainX_tensor = torch.FloatTensor(train_X.values)
trainY_tensor = torch.FloatTensor(train_y.values)

validX_tensor = torch.FloatTensor(valid_X.values)
validY_tensor = torch.FloatTensor(valid_y.values)

testX_tensor = torch.FloatTensor(test_X.values)
testY_tensor = torch.FloatTensor(test_y.values)

dataset = TensorDataset(trainX_tensor, trainY_tensor)
dataset2 = TensorDataset(validX_tensor, validY_tensor)
dataloader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=False,
                        drop_last=False)
dataloader2 = DataLoader(dataset2,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False)

# 설정값
input_dim = len(feature)  # 컬럼 개수
seq_length = 21  # 21일간의 데이터를 바탕으로 예측, 시퀸스 길이
hidden_dim = 100  # 은닉층 LSTM 계층의 output 차원, keras에서 unit 파라미터
output_dim = 1  # 아웃풋 차원
learning_rate = 0.01
nb_epochs = 1000
num_layers = 1  # lstm 계층 개수


class Net(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        # self.fc_1 = nn.Linear(hidden_dim, 32, bias=True)
        # self.fc_2 = nn.Linear(32, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # 학습 초기화를 위한 함수

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim))

    # 예측을 위한 함수
    def forward(self, x):
        x = self.relu(x)
        x, _status = self.lstm(x)
        x = x.view(-1, self.hidden_dim)
        x = self.dropout(x)

        # x = self.relu(x)
        # x = self.fc_1(x)
        # x = self.dropout(x)
        # x = self.relu(x)
        # x = self.fc_2(x)

        x = self.relu(x)
        x = self.fc(x)

        return x


def train_model(model, train_df, valid_df, num_epochs=None, lr=None, verbose=1, patience=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    nb_epochs = num_epochs
    last_epoch = 0

    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)
    val_hist = np.zeros(nb_epochs)
    patience_count = 0
    dfForAccuracy = pd.DataFrame(index=list(range(num_epochs)), columns=['Epoch', 'train_loss', 'valid_loss'])

    for epoch in range(nb_epochs):
        avg_cost = 0
        val_cost = 0
        total_batch = len(train_df)
        total_batch2 = len(valid_df)
        train_bar = tqdm(train_df, file=sys.stdout, colour='green')

        model.train()
        for batch_idx, samples in enumerate(train_bar):
            x_train, y_train = samples
            y_train = y_train.unsqueeze(1)
            x_train, y_train = x_train.to(device), y_train.to(device)

            # seq별 hidden state reset
            model.reset_hidden_state()

            # H(x) 계산
            outputs = model(x_train)

            # cost 계산
            loss = criterion(outputs, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch

        model.eval()
        with torch.no_grad():
            valid_bar = tqdm(valid_df, file=sys.stdout, colour='yellow')
            for batch_idx, samples in enumerate(valid_bar):
                x_valid, y_valid = samples
                y_valid = y_valid.unsqueeze(1)
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                model.reset_hidden_state()
                outputs = model(x_valid)
                val_loss = criterion(outputs, y_valid)
                val_cost += val_loss / total_batch2


        train_hist[epoch] = avg_cost
        val_hist[epoch] = val_cost
        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'train_loss'] = round(avg_cost.item(), 3)
        dfForAccuracy.loc[epoch, 'valid_loss'] = round(val_cost.item(), 3)

        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            print('Epoch:', '%04d' % (epoch), 'val loss :', '{:.4f}'.format(val_cost))
            torch.save(model.state_dict(), f".\\models\\{epoch}.pth")
            dfForAccuracy.to_csv('.\\ModelLoss.csv', index=False)

        # verbose번째 마다 early stopping 여부 확인
        if (epoch % verbose == 0) & (epoch != 0):
            # loss가 커졌다면 early stop
            if val_hist[epoch - verbose] < val_hist[epoch]:
                patience_count += 1
                if patience_count >= patience:
                    print('\n Early Stopping')
                    last_epoch += epoch

                    break

    return model.eval(), train_hist, val_hist, last_epoch


# 모델 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(input_dim, hidden_dim, seq_length, output_dim, num_layers).to(device)
model, train_hist, val_hist, last_epoch = train_model(net, dataloader, dataloader2, num_epochs=nb_epochs,
                                            lr=learning_rate, verbose=1, patience=0)

# epoch별 손실값
fig = plt.figure(figsize=(10, 4))
plt.plot(train_hist[:last_epoch], label="Training loss")
plt.plot(val_hist[:last_epoch], label="Validation loss")
plt.legend()
plt.show()

# 모델 테스트
# 예측 테스트

# pth_ls = glob.glob(os.path.join(".\\models", "*.pth"))
# for path in pth_ls:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Net(input_dim, hidden_dim, seq_length, output_dim, num_layers).to(device)
#     model.load_state_dict(torch.load(path), strict=False)
#     model.eval()
#     with torch.no_grad():
#         pred = []
#         for pr in range(len(testX_tensor)):
#             model.reset_hidden_state()
#
#             testX_tensor = testX_tensor.to(device)
#             predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
#             predicted = torch.flatten(predicted).item()
#             pred.append(predicted)
#
#
#
#
#     def MAE(true, pred):
#         return np.mean(np.abs(true - pred))
#
#     print(os.path.basename(path))
#     print(np.array(pred).reshape(-1), testY_tensor.numpy())
#     print('MAE SCORE : ', MAE(np.array(pred).reshape(-1, 1), testY_tensor.numpy()))

path = f'.\\models\\{last_epoch}.pth'
# path = f'.\\models\\1.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_dim, hidden_dim, seq_length, output_dim, num_layers).to(device)
model.load_state_dict(torch.load(path), strict=False)
model.eval()
with torch.no_grad():
    pred = []
    for pr in range(len(plot_tensor)):
        model.reset_hidden_state()
        plot_tensor = plot_tensor.to(device)
        predicted = model(torch.unsqueeze(plot_tensor[pr], 0))
        predicted = torch.flatten(predicted).item()
        pred.append(predicted)




    def MAE(true, pred):
        return np.mean(np.abs(true - pred))

    print(np.array(pred).reshape(-1), testY_tensor.numpy())
    mae = MAE(np.array(pred).reshape(-1, 1), testY_tensor.numpy())
    plt.figure(figsize=(20, 10), dpi=300)
    plt.title('LSTM 예측 결과'+   '   MAE : ' + str(mae)[:7])
    plt.ylabel('가격')
    plt.plot(df_learn['target'].values, alpha = 0.9, label = 'Real')
    plt.plot(np.array(pred).reshape(-1, 1), alpha = 0.6, linestyle = "--", label = 'Predict')
    plt.legend()
    plt.show()
