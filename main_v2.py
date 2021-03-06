import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
from sklearn import linear_model
import os
from multiprocessing import Process, Queue
pd.set_option('display.max_columns', 30)

def round_price(price):
    if price < 10:
        price = round(price, 2)
    elif price < 100:
        price = round(price, 1)
    elif price < 1000:
        price = round(price/5, 0)*5
    elif price < 10000:
        price = round(price/10, 0)*10
    elif price < 100000:
        price = round(price / 50, 0) * 50
    elif price < 1000000:
        price = round(price / 500, 0) * 500
    else:
        price = round(price / 1000, 0) * 1000
    return price

def criteria_updown(df):
    a = df
    a.reset_index(drop=True, inplace=True)
    if len(df) == 1:
        close_rate = 0
        open_rate = 0
        volume_rate = 0
    else:
        close_rate = df.close[1] / df.close[0]
        open_rate = df.open[1] / df.open[0]
        volume_rate = df.volume[1] / df.volume[0]
    if open_rate < 1:
        open_status = 'down'
    elif open_rate > 1:
        open_status = 'up'
    else:
        open_status = 'normal'
    if close_rate < 1:
        close_status = 'down'
    elif close_rate > 1:
        close_status = 'up'
    else:
        close_status = 'normal'

    if volume_rate < 1:
        volume_status = 'down'
    elif volume_rate > 1:
        volume_status = 'up'
    else:
        volume_status = 'normal'
    result = {'volume_status':volume_status,'volume_rate':volume_rate, 'close_rate':close_rate, 'open_status':open_status}
    return result

def period_end_date(intervals):
    start_time = datetime.now()
    if intervals == "month":
        end_time = start_time + timedelta(days=30)
    elif intervals == "days":
        end_time = start_time + timedelta(days=1)
    elif intervals == "minute240":
        end_time = start_time + timedelta(hours=4)
    elif intervals == "minute60":
        end_time = start_time + timedelta(hours=1)
    elif intervals == "minute30":
        end_time = start_time + timedelta(minutes=30)
    elif intervals == "minute10":
        end_time = start_time + timedelta(minutes=10)
    elif intervals == "minute5":
        end_time = start_time + timedelta(minutes=5)
    elif intervals == "minute3":
        end_time = start_time + timedelta(minutes=3)
    elif intervals == "minute1":
        end_time = start_time + timedelta(minutes=1)
    else:
        end_time = start_time + timedelta(minutes=1)
    result = end_time
    return result

def f_coef_macd_confirm(bdf):
    updown_close = f_reg_coef(bdf, 'close')
    updown_rate_high = f_reg_coef(bdf, 'high')
    updown_rate_low = f_reg_coef(bdf, 'low')
    updown_check = (updown_close>0) & (updown_rate_low>0) &  (updown_rate_high>0)
    updown_rate_volume = f_reg_coef(bdf, 'rate_volume')
    avg_close = np.mean(bdf.close)
    tot_volume = sum(bdf.volume)
    over_volume = np.sum(bdf.rate_volume>1)
    return updown_check, updown_rate_volume, avg_close, tot_volume, over_volume

def chart_name(df_1):
    type = df_1.color[0]
    high = df_1.length_high[0]
    middle = df_1.length_mid[0]
    low = df_1.length_low[0]
    if bool(type == 'red') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle): # ????????? ??????
        result = 'longbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle):# ????????? ??????
        result = 'longbody_umbong'
    elif bool(type == 'red') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):#????????? ??????
        result = 'shortbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):#????????? ??????
        result = 'shortbody_umbong'
    elif bool(type == 'blue') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):#?????? ??????
        result = 'doji_cross'
    elif bool(type == 'red') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):#????????? ??????
        result = 'rickshawen_doji'
    elif bool(type == 'blue') & bool(0 < middle * 5 < high) & bool(low == 0):#????????? ??????
        result = 'stone_cross'
    elif bool(type == 'red') & bool(0 < middle * 5 < low) & bool(high == 0):# ???????????? ??????
        result = 'dragonfly_cross'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle == 0): #:??? ???????????? ??????
        result = 'four_price_doji'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle > 0): #????????????
        result = 'pole_yangbong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low == 0) & bool(middle > 0): #????????????
        result = 'pole_umbong'
    elif bool(type == 'red') & bool(high > 0) & bool(low == 0) & bool(middle > 0): #????????? ????????????
        result = 'upper_tail_pole_yangbong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low > 0) & bool(middle > 0):#???????????? ????????????
        result = 'lower_tail_pole_umbong'
    elif bool(type == 'red') & bool(high == 0) & bool(low > 0) & bool(middle > 0):#???????????? ????????????
        result = 'lower_tail_pole_yangbong'
    elif bool(type == 'blue') & bool(high > 0) & bool(low == 0) & bool(middle > 0):#????????? ????????????
        result = 'upper_tail_pole_umbong'
    elif bool(type == 'blue') & bool(middle * 5 < high) & bool(middle * 5 < low) & bool(middle>0):# ????????? ??????
        result = 'spinning_tops'
    elif bool(type == 'blue') & bool(0 < high <= middle) & bool(0 < low <= middle):# ?????? ??????
        result = 'start'
    else:
        result = 'need to name'
    return result

def box_create(df_1) :
    df_1 = df_1.reset_index(drop=True)
    if (df_1.open[0] <= df_1.close[0]):
        df_1['color'] = 'red'
    else:
        df_1['color'] = 'blue'
    return df_1

def box_information(df):
    default_res = []
    for i in range(len(df)):
        df_1 = df[i:i+1]
        if i == 0:
            default_res = box_create(df_1)
        else:
            next_res = box_create(df_1)
            default_res = pd.concat([default_res, next_res], axis=0)
            default_res.reset_index(drop=True, inplace=True)
    volume = f_vol(df)
    p =pd.DataFrame(volume, columns=['vol_rate'])
    default_res['rate_volume'] = p
    return default_res

def f_reg_coef(df,name):
    #name = 'open'
    df = df[-5:].reset_index(drop=True)
    y_bol_1 = df[name]
    x_arr = []
    y_arr_bol_1 = []
    for i in range(len(df)):  # i : row
        res = [i+1]
        x_arr.append(res)
        res = y_bol_1[i:i + 1].to_list()
        y_arr_bol_1.append(res)

    reg = linear_model.LinearRegression()
    reg.fit(x_arr, y_arr_bol_1)
    val_1 = np.sum(reg.coef_.astype(float))
    return val_1
def f_rsi(df):
    a = df.iloc[:,3] - df.iloc[:,0]
    b = np.sum(a[a >= 0])
    c = abs(np.sum(a[a < 0]))
    if (b == 0) & (c == 0):
        rsi = 50
    else:
        rsi =(b / (c + b) * 100)
    return rsi
def f_bol(df):
    df = df[0:len(df)-1]
    bol_median = np.mean(df['close'])
    bol_std = np.std(df['close'])
    env_higher = round(bol_median * (1+len(df)/1000),3)
    env_lower = round(bol_median * (1-len(df)/1000),3)
    return bol_median, bol_std, env_higher, env_lower

def f_macd(df, a, b, c):
    # a = 12
    # b = 26
    # c = 9
    emaa = f_ema(df, a)[-60:]
    emab = f_ema(df, b)[-60:]
    macd = [(i - j)/j for i, j in zip(emaa, emab)][-60:]
    signal = f_macd_signal(macd, c)[-60:]
    oscillator = [i - j for i, j in zip(macd, signal)]
    result = pd.DataFrame([a for a in zip(oscillator,macd, signal)],columns=['oscillator','macd', 'signal'],index = range(len(oscillator)))
    return result

def f_ema(df, length):
    #df = df2
    #length = 12
    sdf = df[0:length].reset_index(drop=True)
    ema = round(np.mean(df.close[0:length]),0)
    n = np.count_nonzero(df.close.to_list())
    sdf = df[length:n-1]
    res = [ema]
    ls = sdf.close.to_list()
    for i in range(np.count_nonzero(ls)-1):
        ema = round(ls[i+1]*2/(length+1) + ema*(1-2/(length+1)),2)
        res = res + [ema]
    return res
def f_macd_signal(macd, length):
    s = macd[0:length]
    signal = round(np.mean(s), 0)
    macd = macd[-60:]
    n = np.count_nonzero(macd)
    res = [signal]
    for i in range(np.count_nonzero(macd)):
        signal = round(macd[i] * 2 / (length + 1) + signal * (1 - 2 / (length + 1)), 3)
        res = res + [signal]
    return res

def f_vol(df):
    v = [0]
    for i in range(len(df)-1):
        res = [round((df.volume[i+1] - df.volume[i])/df.volume[i],2)]
        v = v + res
    return v
def f_oscillator(df):
    oscillator = f_macd(df, 12, 26, 9)  ################# check1
    min_oscillator = np.min(oscillator.oscillator)
    max_oscillator = np.max(oscillator.oscillator)
    coef_oscillator = f_reg_coef(oscillator, 'oscillator')
    coef_macd = f_reg_coef(oscillator, 'macd')
    coef_signal = f_reg_coef(oscillator, 'signal')
    oscillator = oscillator[-1:].reset_index(drop=True).oscillator[0]
    return oscillator, coef_macd, coef_signal, coef_oscillator, min_oscillator, max_oscillator

def check_buy_case(df, min, max):
    df = df[0:len(df)-1]
    df = df.reset_index(drop=True)
    rate = (df.close[4] - df.open[0]) / df.open[0]
    result = False
    temp1 = df[-1:].reset_index(drop=True)
    temp2 = df[-2:].reset_index(drop=True)
    temp3 = df[-3:].reset_index(drop=True)
    temp4 = df[-4:].reset_index(drop=True)
    temp5 = df[-5:].reset_index(drop=True)
    # 3?????? ??????
    #case1 : ????????? ?????? & ?????? ??? ?????? ??????
    criteria1_1 = np.min(temp1.high[0]-temp1.low[0])/temp1.low[0]
    criteria1_2 = np.min(temp2.high[0]-temp2.low[0])/temp2.low[0]
    criteria1_3 = np.min(temp3.high[0]-temp3.low[0])/temp3.low[0]
    criteria1_4 = np.min(temp4.high[0]-temp4.low[0])/temp4.low[0]
    criteria1_5 = np.min(temp5.high[0]-temp5.low[0])/temp5.low[0]
    criteria1 = (criteria1_1 <= criteria1_2 <= criteria1_3 <= criteria1_4 <= criteria1_5) & (abs(rate) < 0.02)

    #case2 : ????????? ?????? vs ?????? ?????????
    criteria2_1 = (temp1.high[0]-temp1.open[0] == 0) & (temp1.low[0] < np.min(df.low)) & (temp1.color[0] == 'blue')
    criteria2_2 = (temp2.high[0] < max) & ((temp2.open[0]-temp2.low[0])/temp2.low[0] < abs(0.001)) & (temp2.color[0] == 'red')
    criteria2 = criteria2_1 & criteria2_2 & (np.sum(df[-5:].color == "red")>=2) & (np.sum(df[-5:].color == "blue")>=2)

    #case3 : ????????? ?????? & ??????
    criteria3_1 = ((temp1.open[0]-temp1.close[0]) >= (temp1.close[0] - temp1.low[0])*2) * (temp2.color[0] == 'blue')
    criteria3_2 = temp1.high[0] >= temp1.high[0]
    criteria3_3 = np.sum(df[-4:].color == 'blue') == 4
    criteria3_4 = (temp1.high[0] - temp1.open[0])*1.3 >= (temp1.close[0] - temp1.low[0])
    criteria3 = criteria3_1 & criteria3_2 & criteria3_3 & criteria3_4

    #case4 : ????????? ?????? & ?????? ?????? - 2%
    criteria4_1 = (np.max(df[-5:].high) > max*0.95) & (np.min(df[-5:].open) <= min)
    criteria4_2 = ((temp1.close[0] - temp1.low[0])/temp1.low[0] >= 0.005) & ((temp1.open[0] - temp1.close[0])/temp1.close[0] >= 0.005) & (temp1.high[0] <= temp2.close[0])
    criteria4 = criteria4_1 & criteria4_2

    #case5 : ????????? ?????? & ?????? ??????
    criteria5_1 = (temp1.high[0] >= temp2.high[0]) & ((temp1.high[0] - temp1.open[0]) >= (temp1.close[0] - temp1.low[0])*1.5) & ((temp1.open[0] - temp1.close[0])*1.5 <= (temp1.close[0] - temp1.low[0]))
    criteria5_2 = ((temp2.high[0] - temp2.open[0]) >= (temp2.open[0] - temp2.close[0]) * 2) & (temp2.low[0] > temp1.close[0])
    criteria5_3 = (np.sum(df[-4:].color=='blue')==4) & (temp5.color[0]=='red') & ((temp5.open[0]-temp5.close[0])/temp5.close[0] <= 0.0001)
    criteria5 = criteria5_1 & criteria5_2 & criteria5_3

    #case6 : ???????????? & ??????
    criteria6_1 = (temp1.high[0] > temp2.high[0]) & (temp2.close[0] == temp1.open[0]) & (temp1.close[0] <= temp2.low[0]) & ((temp1.close[0] - temp1.low[0])*1.5 <= (temp1.high[0] - temp1.open[0]))
    criteria6_2 = (temp3.open[0] == temp3.high[0]) & (temp3.color[0] == 'blue')
    criteria6_3 = (temp4.open[0] == temp4.close[0]) & (temp4.color[0] == 'blue')
    criteria6_4 = (temp5.open[0] == temp5.close[0]) & (temp5.color[0] == 'blue')
    criteria6 = criteria6_1 & criteria6_2 & criteria6_3 & criteria6_4

    #case7 : ?????? ??????
    criteria7_1 = (temp1.open[0] == temp2.low[0]) & (temp1.open[0]==temp1.close[0])
    criteria7_2 = (temp2.open[0] == temp2.high[0]) & ((temp2.open[0]-temp2.close[0])*1.5 <= (temp2.close[0]-temp2.low[0]))
    criteria7_3 = np.sum(df[-5:].color == "blue") >= 4
    criteria7 = criteria7_1 & criteria7_2 & criteria7_3
    if (criteria1 | criteria2 | criteria3 | criteria4 | criteria5 | criteria6| criteria7):
        result = True
    return result

def check_sell_case(df,max,min):
    df = df.reset_index(drop=True)
    rate = (df.close[4] - df.open[0]) / df.open[0]
    result = False
    temp1 = df[-1:].reset_index(drop=True)
    temp2 = df[-2:].reset_index(drop=True)
    temp3 = df[-3:].reset_index(drop=True)
    temp4 = df[-4:].reset_index(drop=True)
    temp5 = df[-5:].reset_index(drop=True)
    #3?????????
    #????????? ?????? ??? ?????? ??????
    criteria1_1 = (rate >= 0.02) & (temp1.high[0]==temp1.close[0]) & (temp1.low[0] <= temp2.open[0])
    criteria1_2 = (temp2.open[0] == temp2.low[0]) & (temp2.high[0] > temp1.high[0])
    criteria1_3 = (temp3.close[0] < temp2.open[0]) & (temp4.close[0] < temp3.open[0]) & (temp5.close[0] < temp4.open[0])
    criteria1 = criteria1_1 & criteria1_2 & criteria1_3

    # ???????????? ?????? ??? ??????
    criteria2_1 = (rate >= 0.02) & (temp1.high[0] > max)
    criteria2_2 = (temp2.high[0] < temp1.close[0]) & (temp2.open[0] == temp2.low[0])
    criteria2_3 = temp3.high[0] < temp2.close[0]
    criteria2 = criteria2_1 & criteria2_2 & criteria2_3

    # ???????????? ?????? ??? ??????2 (????????????)
    criteria3_1 = (rate >= 0.02) & (temp1.close[0] == temp1.high[0]) & ((temp1.open[0]-temp1.low[0])/temp1.low[0] >= 0.01)
    criteria3_2 = (temp2.open[0] == temp2.close[0]) & (temp2.open[0] < temp1.low[0])
    criteria3 = criteria3_1 & criteria3_2

    # ?????? ?????? ??? ??????3
    criteria4_1 = (temp1.open[0] == temp2.high[0]) & (temp1.color[0] == 'blue') & ((temp1.high[0] - temp1.open[0])/temp1.open[0] >=0.02)
    criteria4_2 = (temp2.close[0] - temp2.open[0])/temp2.open[0] >= 0.02
    criteria4 = criteria4_1 & criteria4_2

    ## 1,2?????? ?????????
    criteria5_1 = (temp1.close[0] == temp1.high[0]) & (((temp1.open[0]-temp1.low[0])/temp1.low[0]) >= ((temp1.close[0] - temp1.open[0])/temp1.open[0]))
    criteria5_2 = (temp2.close[0] == temp2.high[0]) & (((temp2.open[0]-temp2.low[0])/temp2.low[0]) >= ((temp2.close[0] - temp2.open[0])/temp2.open[0]))
    criteria5_3 = np.sum(df[-5:].color=='red') >= 3
    criteria5 = criteria5_1 & criteria5_2 & criteria5_3

    # 2,3?????? ?????????
    criteria6_1 = (temp1.high[0] <= temp2.close[0]) & (temp1.low[0] > min)
    criteria6_2 = (((temp2.open[0]-temp2.low[0])/temp2.low[0]) >= ((temp2.close[0] - temp2.open[0])/temp2.open[0]))
    criteria6_3 = (((temp3.open[0]-temp3.low[0])/temp3.low[0]) >= ((temp3.close[0] - temp3.open[0])/temp3.open[0]))
    criteria6 = criteria6_1 & criteria6_2 & criteria6_3

    # 1?????? ?????????
    criteria7_1 = (temp1.close[0] == temp1.high[0]) & (
                ((temp1.open[0] - temp1.low[0]) / temp1.low[0]) >= ((temp1.close[0] - temp1.open[0]) / temp1.open[0]))
    criteria7_2 = (rate>=0.005) & (np.sum(df[-5:].color=='red')>=4)
    criteria7 = criteria7_1 & criteria7_2

    # 1?????? ?????? ?????????
    criteria8_1 = (temp1.open[0] == temp1.close[0]) & (temp1.high[0] - temp1.close[0] < temp1.open[0] - temp1.low[0])
    criteria8_2 = (rate >= 0.01) & (np.sum(df[-5:].color=='red')>=4)
    criteria8 = criteria8_1 & criteria8_2
    if (criteria1 | criteria2 | criteria3| criteria4 | criteria5 | criteria6 | criteria7 | criteria8):
        result = True
    return result
def generate_rate_minute(coin_name, interval):
    #coin_name = 'KRW-ETH'
    #interval = 'minute1'
    df = []
    while len(df)==0:
        df = pyupbit.get_ohlcv(coin_name, interval=interval, count=100)
    #df = df[:-1]
    time.sleep(0.1)
    rsi_df = df[-11:]
    rsi = f_rsi(rsi_df)
    bol_median, bol_std, env_higher, env_lower= f_bol(rsi_df)
    oscillator, coef_macd, coef_signal, coef_oscillator, min_oscillator, max_oscillator = f_oscillator(df)

    box = box_information(df)
    bdf = box[-11:].reset_index(drop=True)
    cur_price = get_currency(coin_name)

    updown_check, updown_rate_volume, avg_close, tot_volume, over_volume = f_coef_macd_confirm(bdf)  ### check3
    benefit = bol_std / bol_median / 2
    position = 0.5 + (cur_price - bol_median) / bol_median
    tot_trade = avg_close * tot_volume
    min = bol_median - 1.5 * bol_std
    max = bol_median + 1.5 * bol_std

    buy_check = check_buy_case(bdf, min, max)
    buy_check = buy_check   # type1. ?????? ?????? ??????
    buy_check2 = buy_check & (cur_price < bol_median - 2 * bol_std) # ??????????????? ?????? ?????? ??????
    buy_check3 = buy_check & (oscillator <= min_oscillator * 0.8) & (np.sum([(coef_macd > 0), (coef_signal > 0)])>=2) # macd ?????? ??????
    buy_check4 = buy_check & (np.min(df[-20:].close) > bol_median) & (bol_median - 0.1 * bol_std < cur_price < bol_median + 0.1 * bol_std) & updown_check # ???????????????, ?????? ????????? ?????? ??????
    buy_check5 = buy_check & (rsi < 30) # rsi ?????? ??????
    sell_check = check_sell_case(bdf, max,min)
    result = {'coin_name':coin_name, 'interval':interval,
              'cur_price':cur_price, 'benefit':benefit,
              'position':position, 'tot_trade':tot_trade,
              'min_oscillator': min_oscillator, 'max_oscillator': max_oscillator,
                'over_volume': over_volume,
              'buy_check': buy_check, 'sell_check':sell_check,
              'buy_check2':buy_check2, 'buy_check3':buy_check3,
              'buy_check4':buy_check4, 'buy_check5': buy_check5,
              'coef_macd':coef_macd, 'coef_signal':coef_signal,
              'updown_check':updown_check, 'rsi':rsi,
              'avg_close':avg_close, 'tot_volume':tot_volume,
              'oscillator':oscillator, 'coef_oscillator':coef_oscillator,
              'min' : min, 'max':max,
              'bol_median':bol_median, 'bol_std':bol_std,
              'env_higher':env_higher, 'env_lower':env_lower}
    return result
def merge_df(df, directory, name):
    #df = result
    json_file = directory + '/' + name
    # ????????????
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = df.dropna(axis=0)
    df = pd.DataFrame(df)
    if not os.path.isfile(directory):
        try:
            ori_df = pd.read_json(json_file, orient='table')
            df = pd.concat([ori_df, df], axis=0)
            df.reset_index(drop=True, inplace=True)
        except:
            df = df
    else:
        df = df
    if len(df)>0:
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
        df.to_json(json_file, orient='table')
    else:
        try:
            os.remove(json_file)
        except:
            print("??????error: "+ json_file)
    return df


def execute_search_schedule(search_intervals, sec):
    while True:
        try:
            os.remove('temp/temp.json')
        except:
            pass

        tickers = pyupbit.get_tickers(fiat="KRW")
        #coin_name = "KRW-SRM"
        for coin_name in tickers:
            res = []
            # search_intervals = ['minute1','minute3','minute5','minute10', 'minute15', 'minute30','minute60', 'minute240', 'day']
            for interval in search_intervals:
                #coin_name = tickers[0]
                #interval = search_intervals[0]
                try:
                    df = generate_rate_minute(coin_name, interval)
                    res.append(df)
                    time.sleep(0.1)
                except:
                    pass
            df_1 = pd.DataFrame(res)
            merge_df(df_1, 'temp', 'temp.json')
            #print(str(datetime.now()) + "    ?????? ????????? ????????? ??????: " + ', '.join(df_2.coin_name))
        load = []
        load = load_df('temp', 'temp.json')
        if len(load) > 0:
            tot_trade = np.sum(load['tot_trade'])
            load['tot_trade_rate'] = load['tot_trade'] / tot_trade
            load['validation']= (load['benefit'] * (1-load['position'])) * load['tot_trade_rate']

            criteria = load.buy_check & (load.sell_check == False) & (load.over_volume >= 1)
            load1 = load.sort_values('tot_trade', ascending=False).reset_index(drop=True)
            #load2 = load[criteria].sort_values('validation', ascending=False).reset_index(drop=True)
            a = np.min([10, len(load1)])
            load = load1[:a]
            try:
                os.remove('temp/temp.json')
            except:
                print("????????? ?????? ?????? ??????")
            try:
                os.remove('buy_selection_coin/buy_selection_coin.json')
            except:
                pass
            merge_df(load, 'buy_selection_coin', 'buy_selection_coin.json')
            time.sleep(sec)

def load_df(directory, name):
    json_file = directory + '/' + name
    ori_df = []
    if os.path.exists(json_file):
        ori_df = pd.read_json(json_file, orient='table')
    return ori_df

def reservation_cancel(upbit, reserv_list):
    if len(reserv_list) > 0:
        uuids = reserv_list
        while len(uuids) > 0:
            for uuid in uuids:
                try:
                    res = upbit.cancel_order(uuid)
                    uuids.remove(uuid)
                    time.sleep(0.5)
                except:
                    print("?????? ?????? : ??????")
        print("?????? ?????? ?????? : ??????")
    else:
        pass

def coin_buy_price(coin_name):
    #coin_name = 'KRW-BTC'
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))
    # ?????? > ????????? ????????? ??????
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(df_orderbook.buying_YN.to_list()) if value == False]
    no = 0
    if len(check) > 0:
        x = np.max([np.min(check)-1, 0])
        no = np.min([x,14])
    price = df_orderbook.bid_price[no]
    return price
def f_my_coin(upbit):
    df = pd.DataFrame(upbit.get_balances())
    time.sleep(0.1)
    df.reset_index(drop=True, inplace=True)
    df['coin_name'] = df.unit_currency + '-' + df.currency
    df['buy_price'] = pd.to_numeric(df.balance, errors='coerce') * pd.to_numeric(df.avg_buy_price, errors='coerce')
    df = df[df.buy_price >= 5000]
    df.reset_index(drop=True, inplace=True)
    return df
def excute_buy(upbit, coin_name, price, count):
    res = {'uuid':''}
    price = round_price(price)

    if price * count <= 5000:
        pass
    else:
        try:
            res = upbit.buy_limit_order(coin_name, price, count)
            if len(res) > 2:
                print("*******************")
                print("*** ?????? ?????? ??????" + str(res))
        except:
            print("?????? ??????")
    return res
def coin_buy_price(coin_name):
    #coin_name = 'KRW-BTC'
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))
    # ?????? > ????????? ????????? ??????
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(df_orderbook.buying_YN.to_list()) if value == False]
    no = 0
    if len(check) > 0:
        no = min(max(np.min(check)-1, 0),14)
    price = df_orderbook.bid_price[no]
    return price

def auto_buy(upbit, coin_name, investment):
    #coin_name = 'KRW-ADA'
    #coin_name = 'KRW-BTC'
    #df = generate_rate(coin_name, intervals, lines)
    price = pyupbit.get_current_price(coin_name)

    res = {'uuid': ''}
    interval = "minute3"
    df = pd.DataFrame(generate_rate_minute(coin_name, interval), index=[0])
    criteria1 = df.buy_check[0]
    criteria2 = df.buy_check2[0]
    criteria3 = df.buy_check3[0]
    criteria4 = df.buy_check4[0]
    criteria5 = df.buy_check5[0]

    type = "????????????"
    price = coin_buy_price(coin_name)
    if criteria1 :
        type = "CHART"
    elif criteria2:
        type = "(???)??????"
    elif criteria3:
        type = "MACD"
    elif criteria4:
        type = "(???)??????"
    elif criteria5:
        type = "RSI"
    else:
        pass

    if type != "????????????":
        if my_coin_check(upbit, coin_name):
            pass
        else:
            if (df.bol_median[0] - 2 * df.bol_std[0] > price):
                count = investment / price
                res = excute_buy(upbit, coin_name, price, count)
            else:
                pass
        print(str(datetime.now()) + "    (" + type + ") ??????: " + coin_name)
        print("price: " + str(
        round(price, 2)) + "(" + str(round(df['min'][0], 2)) + " ~ " + str(round(df['max'][0], 2))
          + ") / cri1: " + str(criteria1) + " / cri2: " + str(criteria2) + " / cri3: " + str(
        criteria3) + " / cri4: " + str(criteria4) + " / cri5: " + str(criteria5))

    return res
def get_currency(coin_name):
    cur_price = []
    while len(cur_price) == 0:
        try:
            cur_price = [pyupbit.get_current_price(coin_name)]
        except:
            print(coin_name +"??? ????????? ?????? ??????")
        time.sleep(0.1)
    cur_price = cur_price[0]
    return cur_price
def get_money(upbit):
    money = []
    while len(money) == 0:
        try:
            money = [float(pd.DataFrame(upbit.get_balances())['balance'][0])]
        except:
            pass
    money = money[0]
    return money
def execute_buy_schedule(access_key, secret_key, tickers, max_investment):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        money = get_money(upbit)
        investment = np.min([money * 0.7, max_investment])
        while money > 5000:
            reserv_list = []
            if len(tickers)==0:
                load = []
                while len(load) == 0:
                    load = load_df('buy_selection_coin', 'buy_selection_coin.json')
                    time.sleep(1)
                load = load.sort_values('tot_volume', ascending=False).reset_index()
                tickers = list(set(load.coin_name))
            print("????????? ??????: "+ str(tickers))
            res = []
            for coin_name in tickers:
                #coin_name = tickers[0]
                res = auto_buy(upbit, coin_name, investment)
                uuid = res.get('uuid')
                if uuid != '':
                    reserv_list.append(uuid)
                time.sleep(0.1)
            time.sleep(1)
            print("?????? ??????????????????.")
            reservation_cancel(upbit, reserv_list)

def execute_sell(upbit, balance, coin_name, price):
    res = {'uuid': ''}
    price = round_price(price)
    try:
        res = upbit.sell_limit_order(coin_name, price, balance)
        if len(res) > 2:
            print("***?????? ?????? ??????: "+str(res))
    except:
        pass
    return res

def coin_sell_price(coin_name):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # ?????? > ????????? ????????? ??????
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size > x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(df_orderbook.buying_YN.to_list()) if value == False]
    if len(check) > 0:
        no = np.min([np.max([np.min(check) - 1, 0]), 14])
    price = df_orderbook.ask_price[no]
    return price
def my_coin_check(upbit, coin_name):
    my = f_my_coin(upbit)
    check = False
    if (coin_name in list(my.coin_name)):
        my_df = my[my.coin_name == coin_name].reset_index(drop=True)
        avg_price = my_df.avg_buy_price.astype(float)[0]
        balance = my_df.balance.astype(float)[0]
        price = get_currency_price(coin_name)
        if (price < avg_price):
            if (balance * avg_price < 1000000):
                check = True
            else:
                check = False
    return check
def get_currency_price(coin_name):
    cur_price = []
    while len(cur_price) == 0:
        cur_price = [pyupbit.get_current_price(coin_name)]
        time.sleep(0.1)
    cur_price = cur_price[0]
    return cur_price

def auto_sell(upbit, coin_name, cur_price, df, max_benefit, min_benefit, cut_off):
    #coin_name = sell_coin_list[0]
    #coin_name = 'KRW-CBK'
    balance = float(df.balance[0])
    avg_price = float(df.avg_buy_price[0])
    ratio = round((cur_price - avg_price) / avg_price, 3)

    price = cur_price

    res = {'uuid': ''}
    type = "????????????"
    interval = "minute3"
    df = pd.DataFrame(generate_rate_minute(coin_name, interval), index=[0])
    cut_off = - abs(cut_off)
    criteria1 = (ratio >= min_benefit) & (not df.sell_check[0])
    criteria2 = (ratio >= min_benefit) & (df.sell_check[0])
    criteria3 = (ratio <= cut_off)
    criteria4 = (ratio >= max_benefit) & (not df.sell_check[0])
    criteria5 = (ratio >= max_benefit) & (df.sell_check[0])
    criteria6 = (cut_off < ratio < 0) & (df.updown_check[0]) & (df.coef_oscillator[0] > 0) & (not df.sell_check[0]) \
                & (df.buy_check2[0] | df.buy_check3[0] | df.buy_check4[0] | df.buy_check5[0])

    if criteria4:
        type = '??????(??????)'
        upbit.sell_market_order(coin_name, balance)
    elif criteria5:
        type = '??????(??????)'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria1:
        type = '??????1'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria2:
        type = '??????2'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria6:
        investment = avg_price * balance / 5
        count = investment / price
        if my_coin_check(upbit, coin_name):
            type = '????????????'
            print(str(datetime.now()) + "    ("+type+") ??????: " + coin_name + "    min_benefit: "+str(min_benefit))
        else:
            type = '??????'
            price = coin_buy_price(coin_name)
            res = excute_buy(upbit, coin_name, price, count)
    elif criteria3:
        type = '??????'
        upbit.sell_market_order(coin_name, balance)

    if type != "????????????":
        print(str(datetime.now()) + "    ("+type+") ??????: " + coin_name + "    min_benefit: "+str(min_benefit))
    else:
        print(str(datetime.now()) + "    ?????? ??????: " + coin_name)
    return res
def execute_sell_schedule(access_key, secret_key, max_benefit, min_benefit, cut_off):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        my = []
        while len(my)==0:
            my = f_my_coin(upbit)
            time.sleep(0.1)
        tickers = (my.coin_name).to_list()
        reserv_list = []
        res = []
        for coin_name in tickers:
            #coin_name = tickers[0]
            try:
                df = my[my.coin_name == coin_name].reset_index(drop=True)
                cur_price = get_currency_price(coin_name)
                if cur_price * float(df.balance[0]) >= 5000:
                    res = auto_sell(upbit, coin_name, cur_price, df, max_benefit, min_benefit, cut_off)
                else:
                    pass
                    upbit.buy_market_order(coin_name, 5000)
                    time.sleep(0.1)
                    my = f_my_coin(upbit)
                    df = my[my.coin_name == coin_name].reset_index(drop=True)
                    upbit.sell_market_order(coin_name, df.balance[0])
                uuid = res.get('uuid')
                if uuid != '':
                    reserv_list.append(uuid)
                time.sleep(0.1)
            except:
                print(coin_name + "??? ????????????")
        time.sleep(0.1)
        reservation_cancel(upbit, reserv_list)

def coin_trade(access_key, secret_key, tickers, max_investment, max_benefit, min_benefit, cut_off, search_intervals, sec):
    th3 = Process(target=execute_search_schedule, args=(search_intervals, sec))
    th1 = Process(target=execute_buy_schedule, args=(access_key, secret_key, tickers, max_investment))
    th2 = Process(target=execute_sell_schedule, args=(access_key, secret_key, max_benefit, min_benefit, cut_off))
    result = Queue()
    th1.start()
    th2.start()
    th3.start()
    th1.join()
    th2.join()
    th3.join()

if __name__ == '__main__':
    access_key = '5RsZuqMZ6T0tfyjNbIsNlKQ8LI4IVwLaYMBXiaa2'  # ''
    secret_key = 'zPKA1zJwymHMvUSQ2SqYWDgkxNgVfG7Z5jiNLcaJ'  # ''
    # access_key = 'nXlRFXhEdrt9yDmcMGzFxYxSXgTwWUlxcWtlfYhE'  # ''
    # secret_key = 'kV5OBdD43uI8wf3UCEx29Tgu1Yxl6ikxaR6NsN2P'  # ''
    max_investment = 500000
    sell_interval = ["minute10"]
    tickers = []
    #tickers = pyupbit.get_tickers(fiat="KRW")
    min_benefit = 0.002
    max_benefit = 0.1
    cut_off = 0.03
    search_intervals = ["minute120"]
    sec = 60
    coin_trade(access_key, secret_key, tickers, max_investment, max_benefit, min_benefit, cut_off, search_intervals, sec)
