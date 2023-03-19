import pandas as pd
import glob
import os
from tqdm import tqdm


path_list = glob.glob(os.path.join('C:\\Users\\fiter\\Downloads\\정형데이터', '*'))
for idx, path1 in tqdm(enumerate(path_list)):
    type = path1.split('\\')[-1]
    pth_ls = glob.glob(os.path.join(path1, '거래물량', '*.csv'))
    pth_ls2 = glob.glob(os.path.join(path1, '총거래가격', '*.csv'))
    for i, path in enumerate(pth_ls):
        if i == 0:
            df = pd.read_csv(path)
            df = df.fillna(0)
            # print(list(df.columns))
            cols = list(df.columns)
            df[f'{type}_거래량(kg)'] = df[f'{cols[1]}'] + df[f'{cols[2]}'] + df[f'{cols[3]}'] + df[f'{cols[4]}'] + df[f'{cols[5]}'] + df[f'{cols[6]}']
            df[f'{type}_거래량(kg)'] = df[f'{type}_거래량(kg)'] * 1000
        else:
            temp = pd.read_csv(path)
            temp = temp.fillna(0)
            temp[f'{type}_거래량(kg)'] = temp[f'{cols[1]}'] + temp[f'{cols[2]}'] + temp[f'{cols[3]}'] + temp[f'{cols[4]}'] + temp[f'{cols[5]}'] + temp[f'{cols[6]}']
            temp[f'{type}_거래량(kg)'] = temp[f'{type}_거래량(kg)'] * 1000
            df = pd.concat([df, temp], axis=0)

    df = df.reset_index()
    df = df.iloc[:, 1:]

    # print(df)
    # print(df[df['건고추_거래량'] > 0])

    for i, path in enumerate(pth_ls2):
        if i == 0:
            df2 = pd.read_csv(path)
            df2 = df2.fillna(0)
        else:
            temp = pd.read_csv(path)
            temp = temp.fillna(0)
            df2 = pd.concat([df2, temp], axis=0)
    df2 = df2.reset_index()
    df2 = df2.iloc[:, 1:]
    df0 = pd.merge(df, df2, how='inner', on='일자')
    cols0 = list(df0.columns)
    df0[f'{type}_가격(원/kg)'] = (df0[f'{cols0[1]}'] * df0[f'{cols0[8]}'] + \
        df0[f'{cols0[2]}'] * df0[f'{cols0[9]}'] + df0[f'{cols0[3]}'] * \
        df0[f'{cols0[10]}'] + df0[f'{cols0[4]}'] * df0[f'{cols0[11]}'] + \
        df0[f'{cols0[5]}'] * df0[f'{cols0[12]}'] + df0[f'{cols0[6]}'] * \
        df0[f'{cols0[13]}']) / df0[f'{type}_거래량(kg)'] * 1000
    # print(df0)

    if idx == 0:
        fn_df = df0[['일자', f'{type}_거래량(kg)', f'{type}_가격(원/kg)']]
    else:
        plus = df0[['일자', f'{type}_거래량(kg)', f'{type}_가격(원/kg)']]
        fn_df = pd.merge(fn_df, plus, how='inner', on='일자')
    # print(fn_df)

fn_df.to_csv('.\\combined_data2.csv', encoding='utf-8-sig')
