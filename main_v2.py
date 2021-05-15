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
    bdf = bdf[-10:]
    updown_rate = f_reg_coef(bdf, 'close')
    avg_close = np.mean(bdf.close)
    tot_volume = sum(bdf.volume)
    return updown_rate, avg_close, tot_volume

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
        result = 'upper_tail_pole_yangbong'
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
    emaa = f_ema(df, a)[-c*2:]
    emab = f_ema(df, b)[-c*2:]
    macd = [(i - j)/i for i, j in zip(emaa, emab)][-c:]
    signal = f_macd_signal(macd, c)[-c:]
    oscillator = [i - j for i, j in zip(macd, signal)]
    return oscillator[-1]

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
    macd = macd[-9:]
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
    oscillator9 = f_macd(df, 12, 26, 9)  ################# check1
    oscillator6 = f_macd(df, 9, 20, 6)  ################# check1
    oscillator3 = f_macd(df, 6, 12, 3)  ################# check1
    osc_df = pd.DataFrame([oscillator9, oscillator6, oscillator3], columns=['oscillator'])
    oscillator = oscillator3
    coef_oscillator = f_reg_coef(osc_df, 'oscillator')
    return oscillator, coef_oscillator
def check_buy_case(df):
    df = df[-5:].reset_index(drop=True)
    idx = len(df)
    result = False
    if bool(np.sum(df.color == 'blue')>=3) & bool(df.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'blue')>=3) & bool(df.chart_name[idx-1] in ['dragonfly_cross', 'upper_tail_pole_yangong']):
        result = True
    elif bool(df.chart_name[idx-1] in ['pole_yangbong','longbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'blue')>=3) & bool(df.chart_name[idx-1] == 'pole_umbong'):
        result = True
    elif bool(np.sum(df.color == 'blue')>=3) & bool(df.chart_name[idx-1] in ['doji_cross','spinning_tops']):
        result = True
    return result
    bdf["coef_oscillator"] = coef_oscillator
    bdf["coef_open"] = coef_open
    bdf["rsi"] = rsi
    bdf["bol_lower1"] = bol_lower1
    bdf["bol_lower15"] = bol_lower15
    bdf["bol_lower2"] = bol_lower2
    bdf["bol_median"] = bol_median
    bdf["bol_higher1"] = bol_higher1
    bdf["bol_higher15"] = bol_higher15
    bdf["bol_higher2"] = bol_higher2
    bdf["env_lower"] = env_lower
    bdf["env_higher"] = env_higher
    #거래량 많고, red 유형
    result = bdf
    return result
def check_sell_case(df):
    df = df[-5:].reset_index(drop=True)
    idx = len(df)
    result = False
    if bool(np.sum(df.color == 'red') >= 3) & bool(
            df.chart_name[idx - 1] in ['stone_cross', 'lower_tail_pole_umbong', 'upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(
            df.chart_name[idx - 2] in ['longbody_yangbong', 'shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx - 1] in ['spinning_tops', 'doji_cross']):
        result = True
    return result
def generate_rate_minute(coin_name, interval):
    #coin_name = 'KRW-ETH'
    #interval = 'minute10'
    df = []
    while len(df)==0:
        df = pyupbit.get_ohlcv(coin_name, interval=interval, count=51)
    df = df[:-1]
    time.sleep(0.1)
    rsi_df = df[-5:]
    rsi = f_rsi(rsi_df)
    bol_median, bol_std, env_higher, env_lower= f_bol(rsi_df)
    oscillator, coef_oscillator = f_oscillator(df)

    box = box_information(df)
    bdf = box[-20:].reset_index(drop=True)
    cur_price = []
    while len(cur_price) == 0:
        try:
            cur_price = [pyupbit.get_current_price(coin_name)]
        except:
            print(coin_name +"의 현재가 조회 실패")
        time.sleep(0.1)
    cur_price = cur_price[0]
    updown_rate, avg_close, tot_volume = f_coef_macd_confirm(bdf)  ### check3
    buy_check = check_buy_case(bdf)
    buy_check2 = (oscillator < 0) & (bdf[-1:].reset_index(drop=True).color[0]=="blue") & (coef_oscillator > 0)
    sell_check = check_sell_case(bdf)
    benefit = bol_std / bol_median / 2
    position = 0.5 + (cur_price - bol_median) / bol_median
    tot_trade = avg_close * tot_volume

    result = {'coin_name':coin_name, 'interval':interval,
              'cur_price':cur_price, 'benefit':benefit,
              'position':position, 'tot_trade':tot_trade,
              'buy_check': buy_check, 'sell_check':sell_check,
              'buy_check2':buy_check2,
              'updown_rate':updown_rate, 'rsi':rsi,
              'avg_close':avg_close, 'tot_volume':tot_volume,
              'oscillator':oscillator, 'coef_oscillator':coef_oscillator,
              'bol_median':bol_median, 'bol_std':bol_std,
              'env_higher':env_higher, 'env_lower':env_lower}
    return result
def merge_df(df, directory, name):
    #df = result
    json_file = directory + '/' + name
    # 폴더생성
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
        os.remove(json_file)
    return df

def execute_search_schedule(search_intervals, sec):
    while True:
        try:
            os.remove('temp/temp.json')
        except:
            print("선택된 구매 코인 없음")
        tickers = pyupbit.get_tickers(fiat="KRW")
        for coin_name in tickers:
            res = []
            # search_intervals = ['minute30']
            for interval in search_intervals:
                #coin_name = tickers[0]
                #interval = intervals[0]
                try:
                    df = generate_rate_minute(coin_name, interval)
                    res.append(df)
                    time.sleep(0.1)
                except:
                    pass
            df_1 = pd.DataFrame(res)
            #if np.sum(df_1.buy_check) >= len(df_1):
            df_2 = df_1[-1:].reset_index(drop=True)
            merge_df(df_2, 'temp', 'temp.json')
                #print(str(datetime.now()) + "    구매 후보로 선정된 코인: " + ', '.join(df_2.coin_name))
        load = []
        while len(load) == 0:
            load = load_df('temp', 'temp.json')
        tot_trade = np.sum(load['tot_trade'])
        load['tot_trade_rate'] = load['tot_trade'] / tot_trade
        load['validation']= (load['benefit'] * (1-load['position'])) * load['tot_trade_rate']

        criteria = (load.sell_check==False) & (load.oscillator < 0) & (load.benefit > 0.002) & (load.coef_oscillator > 0) & (load.updown_rate > 0)
        #load1 = load[criteria].sort_values('tot_trade', ascending=False).reset_index(drop=True)
        load2 = load[criteria].sort_values('coef_oscillator', ascending=False).reset_index(drop=True)
        load = load2[:min(5,len(load2))]
        try:
            os.remove('temp/temp.json')
        except:
            print("선택된 구매 코인 없음")
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
                    print("예약 취소 : 실패")
        print("모든 예약 취소 : 성공")
    else:
        pass
def f_weight(coin_name, price):
    weight = 0.3
    add_weight = 0.1
    df = pd.DataFrame(generate_rate_minute(coin_name, "minute1"), index=[0])
    criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
    if criteria:
        weight = 0.3 + add_weight
        df = pd.DataFrame(generate_rate_minute(coin_name, "minute3"), index=[0])
        criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
        if criteria:
            weight = weight + add_weight
            df = pd.DataFrame(generate_rate_minute(coin_name, "minute5"), index=[0])
            criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
            if criteria:
                weight = weight + add_weight
                df = pd.DataFrame(generate_rate_minute(coin_name, "minute10"), index=[0])
                criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
                if criteria:
                    weight = weight + add_weight
                    df = pd.DataFrame(generate_rate_minute(coin_name, "minute20"), index=[0])
                    criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
                    if criteria:
                        weight = weight + add_weight
                        df = pd.DataFrame(generate_rate_minute(coin_name, "minute30"), index=[0])
                        criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
                        if criteria:
                            weight = weight + add_weight
                            df = pd.DataFrame(generate_rate_minute(coin_name, "minute60"), index=[0])
                            criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
                            if criteria:
                                weight = weight + add_weight
                                df = pd.DataFrame(generate_rate_minute(coin_name, "minute120"), index=[0])
                                criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
                                if criteria:
                                    weight = weight + add_weight
                                    df = pd.DataFrame(generate_rate_minute(coin_name, "minute240"), index=[0])
                                    criteria = (df.bol_median[0] - 1.5 * df.bol_std[0] > price) & (df.buy_check[0])
                                    if criteria:
                                        weight = weight + add_weight
    return df, weight
def coin_buy_price(coin_name):
    #coin_name = 'KRW-BTC'
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))
    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(df_orderbook.buying_YN.to_list()) if value == False]
    no = 0
    if len(check) > 0:
        no = min(max(np.min(check)-1, 0),14)
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
    my = f_my_coin(upbit)
    if (coin_name in my.coin_name):
        my_df = my[my.coin_name == coin_name].reset_index(drop=True)
        avg_price = my_df.avg_buy_price.astype(float)[0]
        if (price < avg_price):
            price = price * 0.95
    price = round_price(price)

    if price * count <= 5000:
        pass
    else:
        try:
            res = upbit.buy_limit_order(coin_name, price, count)
            if len(res) > 2:
                print("*** 구매 요청 성공" + str(res))
        except:
            print("구매 실패")
    return res
def coin_buy_price(coin_name):
    #coin_name = 'KRW-BTC'
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))
    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(df_orderbook.buying_YN.to_list()) if value == False]
    no = 0
    if len(check) > 0:
        no = min(max(np.min(check)-1, 0),14)
    price = df_orderbook.bid_price[no]
    return price

def auto_buy(upbit, coin_name, buy_interval, investment):
    #coin_name = 'KRW-ADA'
    #df = generate_rate(coin_name, intervals, lines)
    price = pyupbit.get_current_price(coin_name)
    df, weight = f_weight(coin_name,price)
    df = pd.DataFrame(generate_rate_minute(coin_name, buy_interval), index=[0])
    criteria1 = df.buy_check[0]
    criteria2 = (df.coef_oscillator[0] > 0) & ((df.buy_check[0]) | (df.buy_check2[0]))
    criteria3 = (df.bol_median[0] - 2 * df.bol_std[0] > price) & ((df.buy_check[0]) | (df.buy_check2[0]))
    res = {'uuid':''}
    type = "구매없음"
    if criteria1 :
        type = "구매1"
        investment = investment * weight
        price = coin_buy_price(coin_name)
    elif criteria2:
        type = "구매2"
        investment = investment * weight
    elif criteria3:
        type = "구매3"
        investment = investment
    else:
        pass
    if type != "구매없음":
        count = investment / price
        res = excute_buy(upbit, coin_name, price, count)
        print(str(datetime.now()) + "    (" + type + ") 코인: " + coin_name)
    return res

def execute_buy_schedule(access_key, secret_key, buy_interval, max_investment):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
        investment = min(money * 0.2, max_investment)
        while money > 5000:
            reserv_list = []
            tickers = []
            load = []
            while len(load) == 0:
                load = load_df('buy_selection_coin', 'buy_selection_coin.json')
            load = load.sort_values('tot_volume', ascending=False).reset_index()
            tickers = list(set(load.coin_name))
            print("선정된 코인: "+ str(tickers))
            res = []
            for coin_name in tickers:
                #coin_name = tickers[0]
                res = auto_buy(upbit, coin_name, buy_interval, investment)
                uuid = res.get('uuid')
                if uuid != '':
                    reserv_list.append(uuid)
                time.sleep(0.1)
            time.sleep(5)
            reservation_cancel(upbit, reserv_list)

def execute_sell(upbit, balance, coin_name, price):
    res = {'uuid': ''}
    price = round_price(price)
    try:
        res = upbit.sell_limit_order(coin_name, price, balance)
        if len(res) > 2:
            print("***판매 요청 성공: "+str(res))
    except:
        pass
    return res
def f_weight_balance(coin_name):
    weight = 0.7
    add_weight = 0.05
    df = pd.DataFrame(generate_rate_minute(coin_name, "minute1"), index=[0])
    if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
        weight = 0.7
        df = pd.DataFrame(generate_rate_minute(coin_name, "minute3"), index=[0])
        if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
            weight = weight + add_weight
            df = pd.DataFrame(generate_rate_minute(coin_name, "minute5"), index=[0])
            if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
                weight = weight + add_weight
                df = pd.DataFrame(generate_rate_minute(coin_name, "minute10"), index=[0])
                if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
                    weight = weight + add_weight
                    df = pd.DataFrame(generate_rate_minute(coin_name, "minute30"), index=[0])
                    if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
                        weight = weight + add_weight
                        df = pd.DataFrame(generate_rate_minute(coin_name, "minute60"), index=[0])
                        if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
                            weight = weight + add_weight
                            df = pd.DataFrame(generate_rate_minute(coin_name, "minute120"), index=[0])
                            if (df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0):
                                weight = weight + add_weight
    return df, weight
def coin_sell_price(coin_name):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size > x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(df_orderbook.buying_YN.to_list()) if value == False]
    if len(check) > 0:
        no = min(max(np.min(check) - 1, 0), 14)
    price = df_orderbook.ask_price[no]
    return price

def auto_sell(upbit, coin_name, sell_interval, df):
    #coin_name = sell_coin_list[0]
    #coin_name = 'KRW-CBK'
    balance = float(df.balance[0])
    avg_price = float(df.avg_buy_price[0])
    cur_price = []
    while len(cur_price) == 0:
        cur_price = [pyupbit.get_current_price(coin_name)]
        time.sleep(0.1)
    cur_price = cur_price[0]
    ratio = round((cur_price - avg_price) / avg_price, 3)
    df, weight = f_weight_balance(coin_name)
    df = pd.DataFrame(generate_rate_minute(coin_name, sell_interval), index=[0])
    min_benefit = max(df.benefit[0] * (1-df.position[0]), 0.002)
    price = cur_price
    if (balance * price) < 10000:
        balance = balance
    else:
        balance = balance * weight

    res = {'uuid': ''}
    type = "판매없음"
    criteria1 = (ratio >= min_benefit) & ((df.coef_oscillator[0] <= 0) | (df.sell_check[0]))
    criteria2 = (ratio >= min_benefit) & ((df.coef_oscillator[0] >= 0) | (not df.sell_check[0]))
    criteria3 = (ratio <= -0.02) & ((df.coef_oscillator[0] <= 0) | (df.sell_check[0]))
    criteria4 = (ratio >= 0.05)
    criteria5 = (-0.02 < ratio < 0) & (df.updown_rate[0] > 0)

    if criteria1:
        type = '판매1'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria2:
        type = '판매2'
        price = coin_sell_price(coin_name)
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria3:
        type = '손절'
        upbit.sell_market_order(coin_name, balance)
    elif criteria4:
        type = '익절'
        upbit.sell_market_order(coin_name, balance)
    elif criteria5:
        type = '추매'
        investment = avg_price * balance / 3
        count = investment / price
        res = excute_buy(upbit, coin_name, price, count)

    print(str(datetime.now()) + "    ("+type+") 코인: " + coin_name + "    min_benefit: "+str(min_benefit))
    return res
def execute_sell_schedule(access_key, secret_key, sell_interval):
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
                res = auto_sell(upbit, coin_name, sell_interval, df)
                uuid = res.get('uuid')
                if uuid != '':
                    reserv_list.append(uuid)
                time.sleep(0.1)
            except:
                print(coin_name + "의 판매에러")
        time.sleep(0.1)
        reservation_cancel(upbit, reserv_list)

def coin_trade(access_key, secret_key, search_intervals, buy_interval, sell_interval, max_investment, sec):
    th3 = Process(target=execute_search_schedule, args=(search_intervals,sec))
    th1 = Process(target=execute_buy_schedule, args=(access_key, secret_key, buy_interval, max_investment))
    th2 = Process(target=execute_sell_schedule, args=(access_key, secret_key, sell_interval))
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
    search_intervals = ["minute30"]
    buy_interval = ["minute10"]
    sell_interval = ["minute3"]
    sec = 60 * 10 # 후보 리셋 시간
    coin_trade(access_key, secret_key, search_intervals, buy_interval, sell_interval, max_investment, sec)
