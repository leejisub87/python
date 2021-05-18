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

def check_buy_case(df, min, cur_price):
    df = df[-5:].reset_index(drop=True)
    a = np.max([df.close[3], df.open[0]])
    rate = (df.close[4] - a) / a
    result = False
    temp1 = df[-1:].reset_index(drop=True)
    temp2 = df[-2:].reset_index(drop=True)
    color_check = np.sum(df.color == "blue") >= 2
    criteria1 = temp1.open[0] >= temp2.low[0]
    criteria2 = temp1.high[0] >= temp2.high[0]
    criteria3 = (temp1.open[0] > temp1.close[0])
    criteria4 = (rate < -0.002) | (cur_price < min)
    if (criteria1 | criteria2 | criteria3) & criteria4 & color_check:
        result = True
    return result, rate

def check_sell_case(df,max, cur_price):
    df = df[-5:].reset_index(drop=True)
    idx = len(df)
    m = np.max([df.close[2], df.close[3]])
    rate = (df.close[4] -  m) / m
    result = False
    temp1 = df[-2:].reset_index(drop=True)
    temp2 = df[-3:].reset_index(drop=True)
    criteria1 = temp1.high[0] >= temp2.close[0]
    criteria2 = temp1.close[0] <= temp2.close[0]
    criteria3 = (temp1.open[0] > temp1.close[0])
    criteria4 = (rate >= 0.01) | (cur_price < max)
    if (criteria1 | criteria2 | criteria3) & criteria4:
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
    rsi_df = df[-10:]
    rsi = f_rsi(rsi_df)
    bol_median, bol_std, env_higher, env_lower= f_bol(rsi_df)
    oscillator, coef_macd, coef_signal, coef_oscillator, min_oscillator, max_oscillator = f_oscillator(df)

    box = box_information(df)
    bdf = box[-10:].reset_index(drop=True)
    cur_price = []
    while len(cur_price) == 0:
        try:
            cur_price = [pyupbit.get_current_price(coin_name)]
        except:
            print(coin_name +"의 현재가 조회 실패")
        time.sleep(0.1)
    cur_price = cur_price[0]
    updown_check, updown_rate_volume, avg_close, tot_volume, over_volume = f_coef_macd_confirm(bdf)  ### check3
    benefit = bol_std / bol_median / 2
    position = 0.5 + (cur_price - bol_median) / bol_median
    tot_trade = avg_close * tot_volume
    min = bol_median - 1.5 * bol_std
    max = bol_median + 1.5 * bol_std

    buy_check, up_rate = check_buy_case(bdf, min, cur_price)
    buy_check = buy_check   # type1. 구매 차트 모형
    buy_check2 = buy_check & (cur_price < bol_median - 2 * bol_std) # 하락장일때 볼벤 하단 구매
    buy_check3 = buy_check & (oscillator <= min_oscillator * 0.8) & (np.sum([(coef_macd > 0), (coef_signal > 0)])>=2) # macd 이용 구매
    buy_check4 = buy_check & (np.min(df[-20:].close) > bol_median) & (bol_median - 0.1 * bol_std < cur_price < bol_median + 0.1 * bol_std) & updown_check # 상승장일때, 볼벤 미디엄 위에 구매
    buy_check5 = buy_check & (rsi < 30) # rsi 기준 구매
    sell_check = check_sell_case(bdf, max, cur_price)
    sell_check2 = (oscillator >= max_oscillator * 0.9) & ((coef_oscillator < 0) | (np.sum([(coef_macd < 0), (coef_signal <0), ])>=2))
    if up_rate < 0:
        benefit = abs(up_rate) * 0.8
    result = {'coin_name':coin_name, 'interval':interval,
              'cur_price':cur_price, 'benefit':benefit,
              'position':position, 'tot_trade':tot_trade,
              'min_oscillator': min_oscillator, 'max_oscillator': max_oscillator,
              'up_rate':up_rate, 'over_volume': over_volume,
              'buy_check': buy_check, 'sell_check':sell_check,
              'sell_check2':sell_check2,
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
        try:
            os.remove(json_file)
        except:
            print("제거error: "+ json_file)
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
            #print(str(datetime.now()) + "    구매 후보로 선정된 코인: " + ', '.join(df_2.coin_name))

        load = []
        load = load_df('temp', 'temp.json')
        if len(load) > 0:
            tot_trade = np.sum(load['tot_trade'])
            load['tot_trade_rate'] = load['tot_trade'] / tot_trade
            load['validation']= (load['benefit'] * (1-load['position'])) * load['tot_trade_rate']

            criteria = ((load.buy_check) | (load.buy_check2) | (load.buy_check3) | (load.buy_check4) | (load.buy_check5)) | ((load.sell_check==False) & (load.sell_check2==False)) & (load.over_volume>=1)
            load1 = \
                load.sort_values('tot_trade', ascending=False).reset_index(drop=True)
            load2 = load[criteria].sort_values('validation', ascending=False).reset_index(drop=True)
            a = np.min([10, len(load2)])
            load = load2[:a]
            try:
                os.remove('temp/temp.json')
            except:
                print("선택된 구매 코인 없음")
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
                    print("예약 취소 : 실패")
        print("모든 예약 취소 : 성공")
    else:
        pass
def buy_interval_check(coin_name, buy_interval ,price):
    df = pd.DataFrame(generate_rate_minute(coin_name, buy_interval), index=[0])
    criteria = df.buy_check[0] | df.buy_check2[0] | df.buy_check3[0] | df.buy_check4[0] | df.buy_check5[0]
    if criteria:
        return True
    else:
        return False

def sell_interval_check(coin_name, buy_interval ,price):
    df = pd.DataFrame(generate_rate_minute(coin_name, buy_interval), index=[0])
    criteria = df.sell_check[0] | df.sell_check2[0]
    if criteria:
        return True
    else:
        return False
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
        no = np.min([np.max([np.min(check)-1, 0]),14])
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
    #coin_name = 'KRW-BTC'
    #df = generate_rate(coin_name, intervals, lines)
    price = pyupbit.get_current_price(coin_name)
    check1, check2, check3, check4 = False, False, False, False
    if buy_interval_check(coin_name, "minute240", price):
        check1 = True
        if buy_interval_check(coin_name, "minute60", price):
            check2 = True
            if buy_interval_check(coin_name, "minute30", price):
                check3 = True
                if buy_interval_check(coin_name, "minute15", price):
                    check4 = True
    check = np.sum([check1, check2, check3, check4])>=4
    print(coin_name + " / " + str(check1) + " / " + str(check2) + " / " + str(check3)+ str(check4))

    res = {'uuid': ''}
    df = pd.DataFrame(generate_rate_minute(coin_name, "minute5"), index=[0])
    if check :
        criteria1 = df.buy_check[0]
        criteria2 = df.buy_check2[0]
        criteria3 = df.buy_check3[0]
        criteria4 = df.buy_check4[0]
        criteria5 = df.buy_check5[0]

        type = "구매없음"
        price = coin_buy_price(coin_name)
        if criteria1 :
            type = "CHART"
        elif criteria2:
            type = "(하)볼벤"
        elif criteria3:
            type = "MACD"
        elif criteria4:
            type = "(상)볼벤"
        elif criteria5:
            type = "RSI"
        else:
            pass

        if type != "구매없음":
            if my_coin_check(upbit, coin_name):
                pass
            else:
                if (df.bol_median[0] - 1.5 * df.bol_std[0] > price):
                    count = investment / price
                    res = excute_buy(upbit, coin_name, price, count)
                else:
                    pass
            print(str(datetime.now()) + "    (" + type + ") 코인: " + coin_name)
            print("rate: " + str(round(df.up_rate[0], 5)) + "/ price: " + str(
            round(price, 2)) + "(" + str(round(df['min'][0], 2)) + " ~ " + str(round(df['max'][0], 2))
              + ") / cri1: " + str(criteria1) + " / cri2: " + str(criteria2) + " / cri3: " + str(
            criteria3) + " / cri4: " + str(criteria4) + " / cri5: " + str(criteria5))
    else:
        print("조건 불만족: "+coin_name+"/" + str(check1) + "/" + str(check2) + "/" + str(check3))
        print(str(df))
    return res

def execute_buy_schedule(access_key, secret_key, tickers, buy_interval, max_investment):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
        investment = np.min([money * 0.7, max_investment])
        while money > 5000:
            reserv_list = []
            if len(tickers)==0:
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
            time.sleep(1)
            print("구매 탐색중입니다.")
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
def f_balance_check(coin_name, sell_interval):
    df = pd.DataFrame(generate_rate_minute(coin_name, sell_interval), index=[0])
    if ((df.oscillator[0] > 0) & (df.coef_oscillator[0] < 0)) | (df.sell_check[0]) | (df.sell_check2[0])| (df.rsi[0] >70):
        return True
    else:
        return False
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
    check1, check2, check3 = False, False, False
    if sell_interval_check(coin_name, "minute60", price):
        check1 = True
        if sell_interval_check(coin_name, "minute15", price):
            check2 = True
            if sell_interval_check(coin_name, "minute3", price):
                check3 = True
    print("판매: " +coin_name + " / " + str(check1) + " / " + str(check2) + " / " + str(check3))
    check = np.sum([check1, check2, check3]) >= 3

    res = {'uuid': ''}
    type = "판매없음"
    if check:
        df = pd.DataFrame(generate_rate_minute(coin_name, "minute1"), index=[0])
        cut_off = -abs(cut_off)
        criteria1 = (ratio >= min_benefit) & (not df.sell_check[0]) & (not df.sell_check2[0])
        criteria2 = (ratio >= min_benefit) & ((df.sell_check[0]) | (df.sell_check2[0]))
        criteria3 = (ratio <= cut_off)
        criteria4 = (ratio >= max_benefit) & ((not df.sell_check[0]) & (not df.sell_check2[0]))
        criteria5 = (ratio >= max_benefit) & ((df.sell_check[0]) | (df.sell_check2[0]))
        criteria6 = (cut_off < ratio < 0) & (df.updown_check[0]) & (df.coef_oscillator[0] > 0) & (not df.sell_check[0]) \
                    & (not df.sell_check2[0]) & (df.buy_check2[0] | df.buy_check3[0] | df.buy_check4[0] | df.buy_check5[0])

    if criteria4:
        type = '익절(시장)'
        upbit.sell_market_order(coin_name, balance)
    elif criteria5:
        type = '익절(호가)'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria1:
        type = '판매1'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria2:
        type = '판매2'
        res = execute_sell(upbit, balance, coin_name, price)
    elif criteria6:
        investment = avg_price * balance / 3
        count = investment / price
        if my_coin_check(upbit, coin_name):
            type = '구매초과'
            print(str(datetime.now()) + "    ("+type+") 코인: " + coin_name + "    min_benefit: "+str(min_benefit))
        else:
            type = '추매'
            price = coin_buy_price(coin_name)
            res = excute_buy(upbit, coin_name, price, count)
    elif criteria3:
        type = '손절'
        upbit.sell_market_order(coin_name, balance)

    if type !="판매없음":
        print(str(datetime.now()) + "    ("+type+") 코인: " + coin_name + "    min_benefit: "+str(min_benefit))
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
                print(coin_name + "의 판매에러")
        time.sleep(0.1)
        reservation_cancel(upbit, reserv_list)

def coin_trade(access_key, secret_key, tickers, buy_interval, max_investment, max_benefit, min_benefit, cut_off, search_intervals, sec):
    th3 = Process(target=execute_search_schedule, args=(search_intervals, sec))
    th1 = Process(target=execute_buy_schedule, args=(access_key, secret_key, tickers, buy_interval, max_investment))
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
    max_investment = 200000
    buy_interval = ["minute3"]
    sell_interval = ["minute1"]
    tickers = []
    #tickers = pyupbit.get_tickers(fiat="KRW")
    min_benefit = 0.002
    max_benefit = 0.03
    cut_off = 0.015
    search_intervals = ["minute30"]
    sec = 60
    coin_trade(access_key, secret_key, tickers, buy_interval, max_investment, max_benefit, min_benefit, cut_off, search_intervals, sec)
