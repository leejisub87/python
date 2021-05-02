import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
import os
import math
from multiprocessing import Process, Queue
pd.set_option('display.max_columns', 20)

#=========================================
# 구매
#===========================================
def merge_df(df, directory, name):
    #df = result
    json_file = directory + '/' + name
    # 폴더생성
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = df.dropna(axis=0)
    df = pd.DataFrame(df)
    if not os.path.isfile(directory):
        ori_df = pd.read_json(json_file, orient='table')
        df = pd.concat([ori_df, df], axis=0)
        df.reset_index(drop=True, inplace=True)
    else:
        df = df
    df.to_json(json_file, orient='table')
    return df
def coin_buy_price(coin_name, buy_point):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(list(df_orderbook.buying_YN)) if value == False]
    if len(check) > 0:
        no = max(np.max(check) - buy_point, 0)
    else:
        no = 14
    price = df_orderbook.bid_price[max(no - 1, 0)]
    return price

#=========================================
# 구매
#===========================================

#탐색
#===========================================

def check_buy_case(df):
    idx = len(df)
    updown = criteria_updown(df)

    if bool(np.sum(updown.volume_status == 'down') >= 3) & bool(np.sum(updown.volume_status == 'down') >= 3):
        status = 'down'
    elif bool(np.sum(updown.volume_status == 'up') >= 3) & bool(np.sum(updown.volume_status == 'up') >= 3):
        status = 'up'

    result = False
    if bool(status == 'down') & bool(df.chart_name[4] == 'shooting_star'):
        result = True
    elif bool(status =='down') & bool(df.chart_name[4] == 'upper_tail_pole_yangong'):
        result = True
    elif bool(status =='down') & bool(df.chart_name[2] == 'lower_tail_pole_yangong') & bool(
            df.chart_name[3] == 'upper_tail_pole_yangong') & bool(df.chart_name[4] == 'lower_tail_pole_yangong'):
        result = True
    elif bool(df.chart_name[3]=='longbody_umbong') & bool(df.chart_name[4] == 'shortbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4] in ['shortbody_yangbong','rickshawen_doji', 'longbody_yangbong']):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[3] == 'longbody_umbong') & bool(
            df.chart_name[4] in ['longbody_yangbong', 'rickshawen_doji']):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(df.chart_name[3] == 'shortbody_yangbong') & bool(
            df.chart_name[4] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(df.chart_name[3] == 'doji_cross') & bool(
            df.chart_name[4] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(
            df.chart_name[4] == 'shortbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(
            df.chart_name[3] == 'longbody_yangbong') & bool(
            df.chart_name[4] == 'shortbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.color == 'blue') >= 4) & bool(df.chart_name[3] == 'shooting_star') & bool(df.chart_name[4] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[3] == 'shooting_star') & bool(df.chart_name[4] == 'pole_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4] in ['longbody_yangbong', 'shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(df.chart_name[4] in ['stone_cross','lower_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(df.chart_name[4] in ['dragonfly_cross', 'upper_tail_pole_yangong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(np.sum(df.chart_name == 'pole_umbong')>=2) & bool(df.chart_name[4] in ['shooting_star','upper_tail_pole_umbong']) :
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(df.chart_name[4] in ['doji_cross', 'spinning_tops']):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong')>=2) & bool(np.sum(df.chart_name == 'longbody_yangbong')>=1) & bool(df.chart_name[4] in ['longbody_umbong']):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong')>=2) & bool(np.sum(df.chart_name == 'shortbody_umbong')>=1):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong')>=2) & bool(np.sum(df.chart_name[4] in ['shooting_star','lower_tail_pole_umbong'])>=1):
        result = True
    return result
def check_sell_case(df):
    idx = len(df)
    updown = criteria_updown(df)
    result = False
    if bool(np.sum(updown.volume_status == 'down') >= 3) & bool(np.sum(updown.volume_status == 'down') >= 3):
        status = 'down'
    elif bool(np.sum(updown.volume_status == 'up') >= 3) & bool(np.sum(updown.volume_status == 'up') >= 3):
        status = 'up'
    ####################################################
    #### 하락 반전형 캔들
    ## 유성형
    if bool(status == 'up') & bool(df.chart_name[4] == 'upper_tail_pole_yangong') & bool(df.color[4] == 'blue'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[4] == 'lower_tail_pole_yangbong') & bool(df.color[4] == 'red'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[4] == 'lower_tail_pole_umbong') & bool(df.color[4] == 'red'):
        result = True
    elif bool(status == 'up') & (bool(df.chart_name[3] in ['doji_cross', 'rickshawen_doji']) | bool(df.chart_name[4] in ['doji_cross', 'rickshawen_doji'])):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[3] == 'shortbody_yangbong') & bool(df.chart_name[4] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[2] == 'upper_tail_pole_umbong') & bool(df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4]=='lower_tail_pole_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(df.chart_name[4] == 'doji_corss'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(df.chart_name[4] == 'upper_tail_pole_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4] == 'shortbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(df.chart_name[3] == 'shortbody_umbong') & bool(df.chart_name[4] == 'longbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(df.chart_name[4] == 'longbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(df.chart_name[4] == 'longbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(df.chart_name[3] == 'shortbody_yangbong') & bool(df.chart_name[4] == 'longbody_umbong'):
        result = True
#########################################
    elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(df.chart_name[3] == 'doji_corss') & bool(df.chart_name[4] == 'longbody_pole_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[2] == 'shortbody_yangbong') & bool(df.chart_name[3] == 'doji_corss') & bool(df.chart_name[4] == 'shortbody_umbong'):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-1] in ['stone_cross', 'lower_tail_pole_umbong', 'upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-2] in ['longbody_yangbong', 'shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-1] in ['spinning_tops', 'doji_cross']):
        result = True
    return result
def criteria_updown(df):
    res = []
    if len(df) == 1:
        close_rate = 0
        open_rate = 0
        volume_rate = 0
    else:
        for i in range(len(df)-1):
            a = df[i:i+2].reset_index(drop=True)
            a.reset_index(drop=True, inplace=True)
            close_rate = df.close[1] / df.close[0]
            open_rate = df.open[1] / df.open[0]
            volume_rate = df.volume[1] / df.volume[0]
            if open_rate < 1:
                open_status = 'down'
            elif open_rate > 1:
                open_status = 'up'
            else:
                open_status = 'normal'
            if volume_rate < 1:
                volume_status = 'down'
            elif volume_rate > 1:
                volume_status = 'up'
            else:
                volume_status = 'normal'
            result = {'volume_status':volume_status, 'volume_rate':volume_rate, 'close_rate':close_rate, 'open_status':open_status}
            res.append(result)
    t = pd.DataFrame(res).reset_index(drop=True)
    return t
def coin_predict(res, benefit):
    df = res[-5:]
    df.reset_index(inplace=True, drop = True)
    default = df[-1:].reset_index()
    v = list(df.rate_volume)
    influence_v = 0
    if bool(v.index(max(v)) in [3,4]):
        influence_v = 1

    min_buy_price = min(df.open[0], df.close[0]) - 1.96 * np.std(df.length_low[0])
    max_buy_price = min(df.open[0], df.close[0]) + 1.96 * np.std(df.length_low[0])
    max_sell_price = max(df.open[0], df.close[0]) - 1.96 * np.std(df.length_low[0])

    min_sell_price = max(df.open[0], df.close[0]) - 1.96 * np.std(df.length_low[0])
    min_sell_price = max(min_sell_price, min_buy_price * (1+benefit))
    remain_price = (min_sell_price - min_buy_price) / min_buy_price

    check_buy = check_buy_case(df)
    check_sell = check_sell_case(df)
    # 상태
    result = {'remain_price': remain_price, 'influence_v': influence_v, 'check_buy': check_buy, 'check_sell': check_sell,
              'min_buy_price': min_buy_price, 'min_sell_price': min_sell_price,
              'max_buy_price': max_buy_price,  'max_sell_price': max_sell_price}
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
def chart_name(df_1):
    type = df_1.color[0]
    high = df_1.length_high[0]
    middle = df_1.length_mid[0]
    low = df_1.length_low[0]
    if bool(type == 'red') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle): # 롱바디 양봉
        result = 'longbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle):# 롱바디 음봉
        result = 'longbody_umbong'
    elif bool(type == 'red') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):#숏바디 양봉
        result = 'shortbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):#숏바디 음봉
        result = 'shortbody_umbong'
    elif bool(type == 'blue') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):#도지 십자
        result = 'doji_cross'
    elif bool(type == 'red') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):#릭쇼멘 도지
        result = 'rickshawen_doji'
    elif bool(type == 'blue') & bool(0 < middle * 5 < high) & bool(low == 0):#비석형 십자
        result = 'stone_cross'
    elif bool(type == 'red') & bool(0 < middle * 5 < low) & bool(high == 0):# 잠자리형 십자
        result = 'dragonfly_cross'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle == 0): #:포 프라이스 도지
        result = 'four_price_doji'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle > 0): #장대양봉
        result = 'pole_yangbong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low == 0) & bool(middle > 0): #장대음봉
        result = 'pole_umbong'
    elif bool(type == 'red') & bool(high > 0) & bool(low == 0) & bool(middle > 0): #윗꼬리 장대양봉
        result = 'upper_tail_pole_yangong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low > 0) & bool(middle > 0):#아랫꼬리 장대음봉
        result = 'lower_tail_pole_umbong'
    elif bool(type == 'red') & bool(high == 0) & bool(low > 0) & bool(middle > 0):#아랫꼬리 장대양봉
        result = 'lower_tail_pole_yangbong'
    elif bool(type == 'blue') & bool(high > 0) & bool(low == 0) & bool(middle > 0):#윗꼬리 장대음봉
        result = 'upper_tail_pole_umbong'
    elif bool(type == 'blue') & bool(middle * 5 < high) & bool(middle * 5 < low) & bool(middle>0):# 스피닝 탑스
        result = 'spinning_tops'
    elif bool(type == 'blue') & bool(0 < high <= middle) & bool(0 < low <= middle):# 별형 스타
        result = 'start'
    elif bool(type == 'blue') & bool(high > middle) & bool(low == 0) & bool(low < middle):  # 역망치 스타
        result = 'shooting_star'
    else:
        result = 'need to name'
    return result
def box_create(df_1) :
    df_1 = df_1.reset_index(drop=True)
    if bool(df_1.open[0] <= df_1.close[0]):
        df_1['color'] = 'red'
    else:
        df_1['color'] = 'blue'
    df_1['length_high'] = df_1.high[0] - max(df_1.close[0], df_1.open[0])
    df_1['length_low'] = min(df_1.close[0], df_1.open[0]) - df_1.low[0]
    df_1['length_mid'] = max(df_1.close[0], df_1.open[0]) - min(df_1.close[0], df_1.open[0])
    df_1['rate_mid'] = np.abs(df_1.close[0] - df_1.open[0]) / (df_1.open[0] + df_1.close[0]) * 2
    df_1['rate_high'] = df_1['length_high'] / (max(df_1.close[0], df_1.open[0]) + df_1.high[0]) * 2
    df_1['rate_low'] = df_1['length_low'] / (min(df_1.close[0], df_1.open[0]) + df_1.low[0]) * 2
    name = chart_name(df_1)
    df_1['chart_name'] = name
    return df_1
def box_vervus(default_res,benefit):
    default_df = default_res[-6:]
    v = []
    for i in range(len(default_df)-1):
        ddf = default_df[i:i+2].reset_index(drop=True)
        odds_volume = ddf.volume[1]/ddf.volume[0]
        v.append(odds_volume)
    odds_volume = pd.DataFrame(v, columns = ['rate_volume'])
    res = pd.concat([default_df[-5:].reset_index(drop=True), odds_volume.reset_index(drop=True)], axis=1)
    result = coin_predict(res, benefit)
    return result
def box_information(df,benefit):
    default_res = []
    for i in range(len(df)):
        df_1 = df[i:i+1]
        if i == 0:
            default_res = box_create(df_1)
        else:
            next_res = box_create(df_1)
            default_res = pd.concat([default_res, next_res], axis=0)
            default_res.reset_index(drop=True, inplace=True)
    result = box_vervus(default_res, benefit)
    return result
def changed_ratio(df, price_currency):
    if len(df) > 0:
        df_1 = df[-2:].reset_index(drop=True)
        changed_ratio0 = (df_1.close[0] - df_1.open[0]) / df_1.open[0]
        changed_ratio1 = (df_1.close[1] - df_1.open[1]) / df_1.open[1]
        result = changed_ratio0 - changed_ratio1
    return result
def coin_information(coin_name, interval, benefit):
    #coin_name = tickers[1]
    price_currency = pyupbit.get_current_price(coin_name)
    df = []
    while len(df)==0 :
        df = pyupbit.get_ohlcv(coin_name, interval=interval, count=10)
        time.sleep(0.3)
    price_changed_ratio = changed_ratio(df, price_currency) # changed_ratio
    box = box_information(df, benefit) ### criteria_boxplot
    start_date_ask = datetime.now()
    end_date_ask = period_end_date(interval)
    start_date_bid = end_date_ask + timedelta(seconds=1)
    end_date_bid = start_date_bid + (end_date_ask - start_date_ask)
    box['start_date_ask'] = start_date_ask
    box['end_date_ask'] = end_date_ask
    box['start_date_bid'] = start_date_bid
    box['end_date_bid'] = end_date_bid
    box['price_changed_ratio'] = price_changed_ratio
    box['interval'] = interval
    box['coin_name'] = coin_name
    result = box
    return result
def coin_information_interval(tickers, interval, benefit):
    li = []
    tickers = list(set(tickers))
    for coin_name in tickers:
        #coin_name = tickers[1]
        res = coin_information(coin_name, interval, benefit)
        li.append(res)
    df = pd.DataFrame(li)
    result = df
    return result
def coin_validation(upbit, interval, benefit):
    try:
        tickrs_df = load_df('schedule', 'schedule.json')
        if len(tickrs_df[tickrs_df.check_buy]) == 0:
            tickers = pyupbit.get_tickers(fiat="KRW")
        else:
            tickers = tickrs_df['coin_name'].tolist()
    except:
        tickers = pyupbit.get_tickers(fiat="KRW")
    st = time.time()
    print("대상 선택 시작 : "+interval)
    df = coin_information_interval(tickers, interval, benefit)
    df.reset_index(drop=True, inplace=True)
    result = merge_df(df, 'schedule', 'schedule.json')
    et = time.time()
    diff = round(et-st, 2)
    print("대상 선택 종료 : " + str(diff)+"초")
    return result

#===========================================
def avg_price(df):
    open = round(np.mean(df['open']),-1)
    close = round(np.mean(df['close']),-1)
    low = round(np.mean(df['low']),-1)
    high = round(np.mean(df['high']),-1)
    volume = round(np.mean(df['volume']),-1)
    result = {'open':open, 'close':close, 'low':low, 'high':high, 'volume':volume}
    return result
def before_point_price(df, point):
    if len(df) >= point:
        open = df['open'][-1-point]
        close = df['close'][-1-point]
        low = df['low'][-1-point]
        high = df['high'][-1-point]
        volume = df['volume'][-1-point]
        result = {'open': open, 'close': close, 'low': low, 'high': high, 'volume': volume}
    else:
        print(str(point)+"는 최대 "+str(len(df))+"을 넘을 수 없습니다.")
        result = 'error'
    return result
def price_criteria_boxplot(df,point):
    bdf = before_point_price(df, point)
    value = bdf['close'] - bdf['open']
    if value >= 0:
        type ="red"
        diff_high = bdf['high'] - bdf['close']
        diff_middle = value
        diff_low = bdf['open'] - bdf['low']
    else:
        type = "blue"
        diff_high = bdf['high'] - bdf['close']
        diff_middle = - value
        diff_low =  bdf['open'] - bdf['low']
    result = {'type':type,'high':diff_high, 'middle':diff_middle, 'low': diff_low}
    return result
def buy_senario_1(df, coin_name):
    coin_name = coin_name
    levels = 0
    dff = df[-7:]
    box = price_criteria_boxplot(dff, 0)
    a = avg_price(df[-1:])['open'].astype(float)
    b = avg_price(df[-3:])['open'].astype(float)
    c = avg_price(df[-5:])['open'].astype(float)
    d = avg_price(df[-7:])['open'].astype(float)
    if a > b:
        criteria = 'up'
    elif a < b:
        criteria = 'down'
    else:
        if a > c:
            criteria = 'up'
        elif a < c:
            criteria = 'down'
        else:
            if a > d:
                criteria = 'up'
            elif a < d:
                criteria = 'down'
            else:
                criteria = 'normal'
                # 시나리오 1. 상승장 & 주가 상승을 예고하는 캔들 활용

    if bool(box['high'] == 0):
        if criteria == 'up':  # 상승장
            if bool(box['middle'] < box['low']) & bool(box['type'] == 'red'):  # 망치형
                levels = -1
                print(coin_name + '의 주가 (소)하락을 예고')
            elif bool(box['type'] == 'red') & bool(box['middle'] == 0) & bool(box['low'] == 0) & bool(
                    box['high'] == 0):
                levels = 0
            elif bool(box['low'] > box['middle'] * 1.5) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] == 0):
                levels = -1
                print(coin_name + '의 주가 (소)하락을 예고')
            elif bool(type == 'red') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = -1
                print(coin_name + '의 주가 (소)하락을 예고')
            elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 10):
                levels = -2
                print(coin_name + '의 주가 (대)하락을 예고')
            elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = -2
                print(coin_name + '의 주가 (대)하락을 예고')

            elif box['middle'] > box['low']:  # up
                if box['type'] == 'red':
                    if box['low'] == 0:
                        levels = 2
                        print(coin_name + '의 주가 (대)상승을 예고')
                    else:
                        levels = 1
                        print(coin_name + '의 주가 (소)상승을 예고')
                else:
                    if box['low'] == 0:
                        levels = 2
                        print(coin_name + '의 주가 (대)상승을 예고')
                    else:
                        levels = 1
                        print(coin_name + '의 주가 (소)상승을 예고')
        elif criteria == 'down':  # 하락장
            if bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 10):
                levels = 2
                print(coin_name + '의 주가 (대)상승을 예고')
            elif bool(box['type'] == 'blue') & bool(box['middle'] == 0) & bool(box['low'] == 0) & bool(
                    box['high'] == 0):
                levels = 0
            elif bool(box['low'] > box['middle'] * 1.5) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] == 0):
                levels = 1
                print(coin_name + '의 주가 (소)상승을 예고')
            elif bool(type == 'red') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = 1
                print(coin_name + '의 주가 (소)상승을 예고')
            elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = 2
                print(coin_name + '의 주가 (대)반등을 예고')
            elif box['middle'] < box['low']:
                if box['type'] == 'red':  # 망치형
                    levels = 2
                    print(coin_name + '의 주가 (큰)반등을 예고')
                else:  # 교수형
                    levels = 1
                    print(coin_name + '의 주가 (소)반등을 예고')
    return levels
def search_coin(coin_name, interval, benefit):
    buy_point = 0
    sell_point = 0
    validation = []
    df = pyupbit.get_ohlcv(coin_name, interval=interval, count=10)
    time.sleep(0.1)
    validation = coin_information(coin_name, interval, benefit)

    df2 = df[-7:]
    levels = buy_senario_1(df2, coin_name)
    if levels > 0:  # 구매
        buy_point = levels
    elif levels < 0:  # 판매
        sell_point = abs(levels)

    if validation.get('check_buy'):
        buy_point = buy_point + 1 + validation.get('influence_v')
    if validation.get('check_sell'):
        sell_point = sell_point + 1 + validation.get('influence_v')
    validation['buy_point'] = buy_point
    validation['sell_point'] = sell_point
    validation['levels'] = levels
    return validation

#===========================================
def load_df(directory, name):
    ori_df = []
    json_file = directory + '/' + name
    try:
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')
    except:
        print("데이터 로드 실패: "+json_file)
    return ori_df
def reservation_cancel(upbit, directory, name):
    #directory = 'sell_list'
    #name = 'sell_list.json'
    df = load_df(directory, name)
    if len(df) > 0:
        uuids = list(set(df.uuid))
        while len(uuids) > 0:
            for uuid in uuids:
                try:
                    res = upbit.cancel_order(uuid)
                    time.sleep(0.3)
                    uuids.remove(uuid)
                except:
                    pass
        print("모든 예약 취소")
    else:
        print("예약 없음")

####
def f_my_coin(upbit):
    df = pd.DataFrame(upbit.get_balances())
    df.reset_index(drop=True, inplace=True)
    df['coin_name'] = df.unit_currency + '-' + df.currency
    df['buy_price'] = pd.to_numeric(df.balance, errors='coerce') * pd.to_numeric(df.avg_buy_price, errors='coerce')
    return df
def f_price(coin_name, sell_point, type):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['selling_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size > x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(list(df_orderbook.selling_YN)) if value == False]
    if type == 1:
        no = sell_point
        price = df_orderbook.ask_price[min(no, 14)]
    else:
        no = min(check) + sell_point
        price = df_orderbook.ask_price[min(no, 14)]
    return price
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
def coin_sell(upbit, coin_name, mdf, cutoff, benefit, sell_point):
    #coin_name = 'KRW-STPT'
    df = mdf
    df = df[df.coin_name == coin_name].reset_index(drop=True)
    count = df.balance.astype(float)[0]
    price_currency = pyupbit.get_current_price(coin_name)
    b = df.avg_buy_price.astype(float)
    cur_rate = (price_currency - b) / b
    if (cur_rate < cutoff)[0]:
        price = price_currency
        type = "손절"
    elif (cur_rate > benefit)[0]:
        price = f_price(coin_name, sell_point, 1)
        type = "익절"
    else:
        price = f_price(coin_name, sell_point, 2)
        type = "판매시도"

    price = round_price(price)
    res = upbit.sell_limit_order(coin_name, price, count)
    time.sleep(0.2)
    if len(res) > 2:
        print("***판매 요청 정보***")
        print("구분: "+type+"/ 코인: "+coin_name +"/ 예상손익률: "+ str(round(cur_rate[0]*100,2)) + "% / 예상손익금: " + str(round(price*cur_rate[0]*count,0)))
        #a = '9a870a96-3fa9-48b4-98bb-545d5f1f5981'
        uuid = list()
        uuid.append(res.get('uuid'))
        #uuid.append(a)
        result = pd.DataFrame(uuid, columns=['uuid'])
        directory = 'sell_list'
        name = 'sell_list.json'
        merge_df(result, directory, name)

def check_money(upbit):
    df = f_my_coin(upbit)
    save_money = np.sum(df.buy_price)
    KRW = (df[0:1].balance).astype(float)
    my_money = KRW + save_money
    return my_money

def check_money_rate(first_money, currency_money):
    now = datetime.now()
    check_ratio = (currency_money - first_money) / first_money
    print(str(now) + "   총 수익률: " + str(round(check_ratio[0] * 100, 2)) + "%")

def f_sell_ticker(df):
    df = df[df.buy_price > 0]
    df = df.sort_values('buy_price', ascending=False)
    df.reset_index(drop=True, inplace=True)
    sell_tickers = list(set(df.coin_name))
    return sell_tickers

def auto_sell(upbit, interval, cutoff, benefit):
    my_coin = f_my_coin(upbit)
    mdf = my_coin[my_coin.buy_price > 5000].sort_values('buy_price', ascending=False).reset_index(drop=True)
    sell_tickers = f_sell_ticker(mdf)

    for coin_name in sell_tickers:
        #coin_name = sell_tickers[0]
        result = search_coin(coin_name, interval, benefit)
        sell_point = result.get('sell_point')
        coin_sell(upbit, coin_name, mdf, cutoff, benefit, sell_point)
def auto_buy(upbit, buy_tickers, investment, check_interval, benefit):
    for coin_name in buy_tickers:
        #interval = 'minute1'
        result = search_coin(coin_name, check_interval, benefit)
        buy_point = result.get('buy_point')
        price = coin_buy_price(coin_name, buy_point)
        count = investment / price
        res = upbit.buy_limit_order(coin_name, price, count)
        time.sleep(0.3)
        if len(res) > 2:
            print("***구매 요청 정보***")
            print("코인: " + coin_name + "/ 가격: " + str(price) + "/ 수량: " + str(count))
            uuid = list()
            uuid.append(res.get('uuid'))
            result = pd.DataFrame(uuid, columns=['uuid']).reset_index(drop=True)
            directory = 'buy_list'
            name = 'buy_list.json'
            merge_df(result, directory, name)

def selection_tickers(interval, benefit):
    #interval = 'day'
    #df = load_df('schedule/schedule.json')
    df = coin_validation(upbit, interval, benefit)
    df = df[df.check_sell==False].sort_values('remain_price', ascending=False).reset_index(drop=True)
    a = df[0:10].coin_name
    b = df[df.check_buy].coin_name
    c = np.unique(b.append(a)).tolist()
    result = list(c[0:max(len(c),9)])
    return result
def trade(upbit, target_interval, check_interval, investment, cutoff, benefit):
    #os.remove('schedule/schedule.json')
    first_money = check_money(upbit)
    #buy_tickers = selection_tickers(target_interval, benefit)
    buy_tickers = load_df('schedule','schedule.json').coin_name
    buy_tickers = list(buy_tickers)
    print("Today 선택 코인: " + str(buy_tickers))
    while True:
        currency_money = check_money(upbit)
        check_money_rate(first_money, currency_money)
        diff = 0
        st = time.time()
        reservation_cancel(upbit, 'sell_list', 'sell_list.json')
        reservation_cancel(upbit, 'buy_list', 'buy_list.json')
        while diff < 30:
            result = Queue()
            th1 = Process(target=auto_buy, args=(upbit, buy_tickers, investment, check_interval, benefit))
            th2 = Process(target=auto_sell, args=(upbit, check_interval, cutoff, benefit))
            th1.start()
            th2.start()
            th1.join()
            th2.join()
            et = time.time()
            diff = et - st
        # init_time(10, auto_buy(upbit, buy_tickers, investment, check_interval), 'buy_list', 'buy_list.json')
        # init_time(10, auto_sell(upbit, check_interval, cutoff, benefit), 'sell_list', 'sell_list.json')


if __name__ == '__main__':
    #intervals = ["day", "minute240", "minute60", "minute30", "minute15", 'minute10', 'minute5']
    access_key = 'ZvHxer7F6MuYNTbBODOtO7L0y6BVhdbhbblRDhXB'
    secret_key = 'wF6x0CPDzwYFfZgI2wnmhKNBr99WmiXe0QWyqxGS'
    upbit = pyupbit.Upbit(access_key, secret_key)
    investment = 5500
    cutoff = -0.009
    benefit = 0.02
    check_interval = 'minute5'
    target_interval = 'day'
    trade(upbit, target_interval, check_interval, investment, cutoff, benefit)
