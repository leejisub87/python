import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
from sklearn import linear_model
import os
from multiprocessing import Process, Queue
pd.set_option('display.max_columns', 20)


def coin_trade(upbit, investment, cutoff, benefit):
    th2 = Process(target=execute_buy_schedule, args=(upbit, intervals, investment))
    th3 = Process(target=execute_sell_schedule, args=(upbit, intervals, cutoff, benefit))
    result = Queue()
    th2.start()
    th3.start()
    th2.join()
    th3.join()
    # auto_sell(upbit, coin_name, intervals, cutoff, benefit)
    # auto_buy(upbit, coin_name, intervals, investment)

    #reservation_cancel(upbit, 'buy_list', 'buy_list.json')
    #reservation_cancel(upbit, 'sell_list', 'sell_list.json')


def execute_buy_schedule(upbit, intervals,investment):
    while True:
        st = time.time()
        diff = 0
        buy_df = []
        reservation_cancel(upbit, 'buy_list', 'buy_list.json')
        while diff < 60:
            money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
            time.sleep(0.5)

            if money <= max(5000, investment):
                print("주문금액 부족")
            else:
                tickers = pyupbit.get_tickers(fiat="KRW")
                for coin_name in tickers:
                    auto_buy(upbit, coin_name, intervals, investment)
                    time.sleep(0.1)
            et = time.time()
            diff = et - st
def execute_sell_schedule(upbit, intervals, cutoff, benefit):
    while True:
        st = time.time()
        diff = 0
        buy_df = []
        sell_df = []
        reservation_cancel(upbit, 'sell_list', 'sell_list.json')
        while diff < 60:
            tickers = pyupbit.get_tickers(fiat="KRW")
            for coin_name in tickers:
                auto_sell(upbit, coin_name, intervals, cutoff, benefit)

def auto_buy(upbit, coin_name, intervals, investment):
    result = []
    for interval in intervals:
        res = generate_rate(coin_name, interval, lines)
        result.append(res)
    df = pd.DataFrame(result)
    validation = check_updown(df)  # dwon시, rate_env = 0에 가까울수록 구매 // up시, rate_env =

    if df.type[0] == 'try_buy':
        if np.sum(df.rate_bol > 1) >= 4:
            if validation == 'up':
                if np.sum(df.next_pattern == 'down') >= 1 & df.next_pattern[0] == 'up':
                    print("예측 상승(약): " + coin_name)
                    no = 4
                elif np.sum(df.next_pattern == 'down') >= 2 & df.next_pattern[0] == 'up':
                    print("예측 상승(중): " + coin_name)
                    no = 2
                elif np.sum(df.next_pattern == 'down') >= 3 & df.next_pattern[0] == 'up':
                    print("예측 상승(강): " + coin_name)
                    no = 0
                price = min(df.currency[0], coin_buy_price(coin_name, no))
                count = investment / price * df.rate_env[0]
                excute_buy(upbit, price, count)
            elif validation == 'down':
                if np.sum(df.next_pattern == ['down', 'down', 'down', 'up']) >= 4:
                    print("예측 반등(강): " + coin_name)
                    no = 0
                elif np.sum(df.next_pattern == 'normal') == 1 & np.sum(df.next_pattern == 'down') == 2 & \
                        df.next_pattern[0] == 'up':
                    print("예측 반등(중): " + coin_name)
                    no = 3
                elif np.sum(df.next_pattern == 'normal') == 2 & np.sum(df.next_pattern == 'down') == 1 & \
                        df.next_pattern[0] == 'up':
                    print("예측 반등(약): " + coin_name)
                    no = 5
            # 구매
            price = min(df.currency[0], coin_buy_price(coin_name, no))
            count = investment / price * df.rate_env[0]
            excute_buy(upbit, price, count)
def auto_sell(upbit, coin_name, intervals, cutoff, benefit):
    result = []
    for interval in intervals:
        res = generate_rate(coin_name, interval, lines)
        result.append(res)
    df = pd.DataFrame(result)
    validation = check_updown(df)  #dwon시, rate_env = 0에 가까울수록 구매 // up시, rate_env =

    my = f_my_coin(upbit)
    my_balance = 0
    if np.sum(my.coin_name == coin_name) > 0:
        my_df = my[my.coin_name == coin_name].reset_index(drop=True)
        my_balance = my_df.balance.astype(float)[0]
        avg_price = my_df.avg_buy_price.astype(float)[0]
        cur_price = df.currency[0]
        ratio = round((cur_price-avg_price)/avg_price,3)
        if ratio < - abs(cutoff) :
            price = df.currency[0]
            balance = my_balance
            execute_sell(upbit, balance, coin_name, price)
        elif ratio > benefit:
            if df.type[0] =='try_sell':
                if np.sum(df.rate_bol > 1) >= 4:
                    if bool(validation == 'up'):
                        if np.sum(df.next_pattern == 'up')>=1 & df.next_pattern[0] == 'down':
                            print("예측 하락(약): " + coin_name)
                            no = 4
                        elif np.sum(df.next_pattern == 'up')>=2 & df.next_pattern[0] == 'down':
                            print("예측 하락(중): " + coin_name)
                            no = 2
                        elif np.sum(df.next_pattern == 'up')>=3 & df.next_pattern[0] == 'down':
                            print("예측 하락(강): " + coin_name)
                            no = 0
                        print('sell')
                        price = min(df.currency[0], coin_sell_price(coin_name, 2))
                        balance = my_balance * df.rate_env[0]
                        execute_sell(upbit, balance, coin_name, price)
        else:
            print("판매 대기:"+coin_name)

def f_my_coin(upbit):
    df = pd.DataFrame(upbit.get_balances())
    df.reset_index(drop=True, inplace=True)
    df['coin_name'] = df.unit_currency + '-' + df.currency
    df['buy_price'] = pd.to_numeric(df.balance, errors='coerce') * pd.to_numeric(df.avg_buy_price,
                                                                                 errors='coerce')
    return df
def excute_buy(upbit, price, count):
    price = round_price(price)
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
def execute_sell(upbit, balance, coin_name, price):
    if price * balance > 5000:
        price = round_price(price)
        try:
            res = upbit.sell_limit_order(coin_name, price, balance)
            time.sleep(0.5)
            if len(res) > 2:
                print("***판매 요청 정보***")
                print("코인: " + coin_name + "/ 가격: " + str(price) + "/ 수량: " + str(count))
                # a = '9a870a96-3fa9-48b4-98bb-545d5f1f5981'
                uuid = list()
                uuid.append(res.get('uuid'))
                # uuid.append(a)
                result = pd.DataFrame(uuid, columns=['uuid'])
                directory = 'sell_list'
                name = 'sell_list.json'
                merge_df(result, directory, name)
        except:
            pass
    else:
        print("판매 실패: " + coin_name + "은 판매 최소금액이 부족")

def coin_buy_price(coin_name,m):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(list(df_orderbook.buying_YN)) if value == False]
    if len(check) > 0:
        no = max(np.max(check), 0)
    else:
        no = m
    price = df_orderbook.bid_price[max(no - 1, 0)]
    return price
def coin_sell_price(coin_name,m):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size > x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(list(df_orderbook.buying_YN)) if value == False]
    if len(check) > 0:
        no = max(np.max(check), 0)
    else:
        no = m
    price = df_orderbook.bid_price[max(no - 1, 0)]
    return price

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
def check_buy_case(df):
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
def check_sell_case(df):
    idx = len(df)
    result = False
    if bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong','upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-2] in ['longbody_yangbong','shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-1] in ['spinning_tops','doji_cross']):
        result = True
    return result
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
def coin_predict(default_df):
    df = default_df[-5:]
    df.reset_index(inplace=True, drop = True)
    default = df[-1:].reset_index()
    check_buy = check_buy_case(df)
    check_sell = check_sell_case(df)
    # 상태
    result = {'check_buy':check_buy,'check_sell': check_sell}
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

def box_vervus(default_res):
    df = default_res[-5:]
    default_df = []
    res = []
    for i in range(len(df)-1):
        a = default_res[i:i+1].reset_index(drop= True)
        b = default_res[i+1:i+2].reset_index(drop= True)
        result = []
        l = ['open', 'high', 'low', 'close', 'volume', 'length_high', 'length_low', 'length_mid', 'rate_mid', 'rate_high', 'rate_low']
        for col_name in l:
            if a[col_name][0] > 0:
                c = (b[col_name][0] - a[col_name][0])/a[col_name][0]
            else:
                c = 0
            result.append({'col_name': col_name, 'value':c})
        df_1 = pd.DataFrame(result)
        columns_name = list('vs_'+df_1.col_name)
        value = list(df_1.value)
        b = b.reset_index(drop=True)
        for j in range(len(value)):
            b[columns_name[j]] = value[j]
        if i == 0 :
            default_df = b
        else:
            default_df = pd.concat([default_df, b], axis=0)
    default_df['avg_length_high'] = np.mean(df.length_high)
    default_df['avg_length_mid'] = np.mean(df.length_mid)
    default_df['avg_length_low'] = np.mean(df.length_low)
    default_df['std_length_high'] = np.std(df.length_high)
    default_df['std_length_mid'] = np.std(df.length_mid)
    default_df['std_length_low'] = np.std(df.length_low)
    result = coin_predict(default_df)
    return result

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
    result = box_vervus(default_res)
    return result

def check_updown(df):
    x = df['latest']
    y_bol = df['rate_bol']
    y_env = df['rate_env']
    x_arr = []
    y_arr_bol = []
    y_arr_env = []
    for i in range(len(df)):  # i : row
        res = x[i:i + 1].to_list()
        x_arr.append(res)
        res = y_bol[i:i + 1].to_list()
        y_arr_bol.append(res)
        res = y_env[i:i + 1].to_list()
        y_arr_env.append(res)

    reg = linear_model.LinearRegression()
    reg.fit(x_arr, y_arr_bol)
    val_1 = reg.coef_
    reg.fit(x_arr, y_arr_env)
    val_2 = reg.coef_
    v1 = np.sum(val_1.astype(float))
    v2 = np.sum(val_2.astype(float))
    if bool(v1 >0) & bool(v2 >0):
        check = 'up'
    elif bool(v1 < 0) & bool(v2 < 0):
        check = 'down'
    else :
        check = 'normal'
    return check

def generate_rate(coin_name, interval, lines):
    currency = pyupbit.get_current_price(coin_name)
    df = pyupbit.get_ohlcv(coin_name, interval = interval, count = lines)
    df = df[0:len(df)-1]
    box = box_information(df)
    df['center'] = np.mean([df['close'],df['open']])
    bol_median = np.mean([df['high'],df['center'] ,df['low']])
    bol_std = np.std([df['high'],df['center'] ,df['low']])
    bol_higher = bol_median + 2 * bol_std
    bol_lower = bol_median - 2 * bol_std
    env_higher = bol_median * 1.03 + 2 * bol_std
    env_lower = bol_median * 0.97 - 2 * bol_std

    value = currency - bol_median
    if value > 0:
        type = 'try_sell'
        rate_bol = (currency - bol_median)/(bol_higher - bol_median)
        rate_env = (currency - bol_median)/(env_higher - bol_median)
    else:
        type = 'try_buy'
        rate_bol = (bol_median - currency)/(bol_median - bol_lower)
        rate_env = (bol_median - currency)/(bol_median - env_lower)

    if rate_env < 0:
        rate_env = 0
    if box.get('check_buy'):
        next_pattern = 'up'
    elif box.get('check_sell'):
        next_pattern = 'down'
    else:
        next_pattern = 'normal'
    idx = intervals.index(interval)
    result = {'coin_name': coin_name, 'currency':currency, 'interval': interval, 'latest': idx, 'type': type, 'next_pattern': next_pattern, 'rate_bol': rate_bol, 'rate_env': rate_env,}
    return result



# input 1번 불러오면 되는 것들
if __name__ == '__main__':
    access_key = 'DXqzxyCQkqL9DHzQEyt8pK5izZXU03Dy2QX2jAhV'  # ''
    secret_key = 'x6ubxLyUVw03W3Lx5bdvAxBGWI7MOMJjblYyjFNo'  # ''
    upbit = pyupbit.Upbit(access_key, secret_key)
    # intervals = ["day", "minute240", "minute60", "minute30", "minute15", 'minute10', 'minute5']
    intervals = ["minute1", "minute5", "minute15", "minute30"]
    tickers = pyupbit.get_tickers(fiat="KRW")
    investment = 50000
    cutoff = 0.009
    benefit = 0.019
    lines = 10

    access_key = 'DXqzxyCQkqL9DHzQEyt8pK5izZXU03Dy2QX2jAhV'  # ''
    secret_key = 'x6ubxLyUVw03W3Lx5bdvAxBGWI7MOMJjblYyjFNo'  # ''
    upbit = pyupbit.Upbit(access_key, secret_key)
    lines = 10
    intervals = ["minute1", "minute5", "minute15", "minute30"]

    coin_trade(upbit, investment, cutoff, benefit)
