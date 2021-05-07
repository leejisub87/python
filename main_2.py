import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
from sklearn import linear_model
import os
from multiprocessing import Process, Queue
pd.set_option('display.max_columns', 20)

def auto_search(upbit, coin_name, intervals, lines):
    #coin_name = 'KRW-TON'
    df = generate_rate(coin_name, intervals, lines)
    price = coin_buy_price(coin_name)
    result = result_print(df)
    result = pd.DataFrame(result, index=[0])
    result['coin_name'] = coin_name
    result['price'] = price
    result['down_confirm_buy'] = down_confirm_buy(df, price)
    result['up_confirm_buy'] = up_confirm_buy(df, price)
    result['down_confirm_sell'] = down_confirm_sell(df, price)
    result['up_confirm_sell'] = up_confirm_sell(df, price)
    return result

def execute_search_schedule(upbit, intervals, lines):
    while True:
        tickers = pyupbit.get_tickers(fiat="KRW")
        res = []
        for coin_name in tickers:
            try:
                result = auto_search(upbit, coin_name, intervals, lines)
                res.append(result)
            except:
                pass


def execute_buy_schedule(upbit, ticekrs, intervals, investment, lines, sec):
    while True:
        if not ticekrs:
            tickers = pyupbit.get_tickers(fiat="KRW")
        st = time.time()
        diff = 0
        buy_df = []
        reservation_cancel(upbit, 'buy_list', 'buy_list.json')
        money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
        while bool(money > investment) | bool(diff <= sec):
            #tickers = selection_ticker(tickers)
            money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
            time.sleep(0.1)
            if money < max(5000, investment):
                pass
            else:
                for coin_name in tickers:
                    #coin_name = tickers[0]
                    try:
                        auto_buy(upbit, coin_name, intervals, investment, lines)
                    except:
                        pass
            et = time.time()
            diff = et - st

def execute_sell_schedule(upbit, intervals, cutoff, benefit, lines):
    while True:
        st = time.time()
        diff = 0
        buy_df = []
        sell_df = []
        reservation_cancel(upbit, 'sell_list', 'sell_list.json')
        while diff < 60:
            my = f_my_coin(upbit)
            time.sleep(0.1)
            tickers = (my.coin_name).to_list()
            if not tickers:
                pass
            else:
                for coin_name in tickers:
                    #coin_name = tickers[0]
                    try:
                        auto_sell(upbit, coin_name, intervals, cutoff, benefit, lines)
                    except:
                        pass
def down_confirm_buy(df,price):
    check_buy = bool(np.sum(df.rate_open > 0) >= 3) \
                & bool(np.sum(df.volume_up > 0) >= 3) \
                & bool(np.sum(df.rate_red_bar > 0.5) >= 3) \
                & bool(np.sum(df.coef_macd > 0) >= 3) \
                & bool(np.sum(abs(df.coef_mid) < 0.1) >= 3) \
                & bool(np.sum(df.rsi < 32) >= 3) \
                & bool(np.sum(df.bol_lower > price) >= 3)
    return check_buy
def up_confirm_buy(df,price):
    check_buy = bool(np.sum(df.rate_open > 0) >= 3) \
                & bool(np.sum(df.volume_up > 0) >= 3) \
                & bool(np.sum(df.rate_red_bar > 0.5) >= 3) \
                & bool(np.sum(df.coef_macd > 0) >= 3) \
                & bool(np.sum(abs(df.coef_mid) > 0) >= 3) \
                & bool(np.sum(df.rsi > 30) >= 3) \
                & bool(np.sum(df.bol_lower < price) >= 3) \
                & bool(np.sum(df.bol_higher > price) >= 3)
    return check_buy
def result_print(dfp):
    result = {'avg_rate_open' : round(f_reg_coef(dfp, 'rate_open'),3)
    ,'avg_volume_up' : round(f_reg_coef(dfp, 'volume_up'),3)
    ,'avg_rate_red_bar' : round(f_reg_coef(dfp, 'rate_red_bar'),3)
    ,'avg_coef_macd' : round(f_reg_coef(dfp, 'coef_macd'),3)
    ,'avg_coef_mid' : round(f_reg_coef(dfp, 'coef_mid'),3)
    ,'avg_bol_higher' : round(np.mean(dfp.bol_higher),3)
    ,'avg_bol_lower' : round(np.mean(dfp.bol_lower),3)
    ,'avg_env_higher' : round(np.mean(dfp.env_higher),3)
    ,'avg_env_lower' : round(np.mean(dfp.env_lower),3)
    ,'avg_rsi' : round(np.mean(dfp.rsi),3)}
    return result
def up_confirm_sell(df,price):
    result = bool(np.sum(df.rate_open < 0) >= 3) \
                & bool(np.sum(df.volume_up > 0) >= 3) \
                & bool(np.sum(df.rate_red_bar < 0.5) >= 3) \
                & bool(np.sum(df.coef_macd < 0) >= 3) \
                & bool(np.sum(df.coef_mid < 0) >= 3) \
                & bool(np.sum(df.bol_higher < price) >= 3) \
                & bool(np.sum(df.rsi > 68) >= 3)
    return result
def down_confirm_sell(df,price):
    result = bool(np.sum(df.rate_open < 0) >= 3) \
                & bool(np.sum(df.volume_up > 0) >= 3) \
                & bool(np.sum(df.rate_red_bar <0.5) >= 3) \
                & bool(np.sum(df.coef_macd < 0) >= 3) \
                & bool(np.sum(abs(df.coef_mid) < 0) >= 3) \
                & bool(np.sum(df.rsi < 68) >= 3) \
                & bool(np.sum(df.bol_lower < price) >= 3) \
                & bool(np.sum(df.bol_higher > price) >= 3)
    return result
def auto_buy(upbit, coin_name, intervals, investment, lines):
    #coin_name = 'KRW-BTT'
    df = generate_rate(coin_name, intervals, lines)
    price = coin_buy_price(coin_name)
    count = investment / price
    result = result_print(df)
    print('coin_name: '+coin_name+'/ price:'+ str(price) + '/ '+ str(result))
    if down_confirm_buy(df, price):
        count = count
        excute_buy(upbit, coin_name, price, count, investment)
    elif up_confirm_buy(df, price):
        count = count * 0.1
        excute_buy(upbit, coin_name, price, count, investment)

def auto_sell(upbit, coin_name, intervals, cutoff, benefit, lines):
    #coin_name = tickers[0]
    #coin_name = 'KRW-UPP'
    df = generate_rate(coin_name, intervals, lines)
    df = pd.DataFrame(df, index=[0])
    price = coin_buy_price(coin_name)
    result = result_print(df)
    res = f_balance(upbit, coin_name)
    balance = res.get('my_balance')
    avg_price = res.get('avg_price')
    cur_price = pyupbit.get_current_price(coin_name)
    ratio = round((cur_price - avg_price) / avg_price, 3)
    price = coin_sell_price(coin_name)
    if ratio < - abs(cutoff):
        price = cur_price
        print('sell(손절)/ coin_name: ' + coin_name + '/ price:' + str(price) + '/ ' + str(result))
        execute_sell(upbit, balance, coin_name, price)
    elif bool(ratio >= benefit):
        if up_confirm_sell(df, price):
            price = cur_price
            balance = balance * 0.8
            print('sell(익절)/ coin_name: ' + coin_name + '/ price:' + str(price) + '/ ' + str(result))
        elif down_confirm_sell(df, price):
            price = cur_price
            balance = balance * 0.8
            print('sell(익절)/ coin_name: ' + coin_name + '/ price:' + str(price) + '/ ' + str(result))
        execute_sell(upbit, balance, coin_name, price)

def excute_buy(upbit, coin_name, price, count, investment):
    price = round_price(price)
    if price * count <=5000:
        pass
    else:
        try:
            res = upbit.buy_limit_order(coin_name, price, count)
        except:
            count = investment / price
            res = upbit.buy_limit_order(coin_name, price, count)
        print("구매")
        print(res)
        time.sleep(0.1)
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
            time.sleep(0.1)
            if len(res) > 2:
                print("***판매 요청 정보***")
                print("코인: " + coin_name + "/ 가격: " + str(price) + "/ 수량: " + str(balance))
                # a = '9a870a96-3fa9-48b4-98bb-545d5f1f5981'
                uuid = list()
                uuid.append(res.get('uuid'))
                result = pd.DataFrame(uuid, columns=['uuid'])
                directory = 'sell_list'
                name = 'sell_list.json'
                merge_df(result, directory, name)
        except:
            pass
    else:
        print("판매 실패: " + coin_name + "은 판매 최소금액이 부족")

def coin_buy_price(coin_name):
    #coin_name = 'KRW-BTC'
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(list(df_orderbook.buying_YN)) if value == False]
    if len(check) > 0:
        no = min(max(np.min(check)-1, 0),14)
    price = df_orderbook.bid_price[no]
    return price

def coin_sell_price(coin_name):
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    time.sleep(0.1)
    df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
    df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

    # 매수 > 매도의 가격을 측정
    df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size > x.cum_bid_size, axis='columns')
    check = [i for i, value in enumerate(list(df_orderbook.buying_YN)) if value == False]
    if len(check) > 0:
        no = min(max(np.min(check) - 1, 0), 14)
    price = df_orderbook.ask_price[0]
    return price

def f_balance(upbit, coin_name):
    my = f_my_coin(upbit)
    my_df = my[my.coin_name == coin_name].reset_index(drop=True)
    my_balance = my_df.balance.astype(float)[0]
    avg_price = my_df.avg_buy_price.astype(float)[0]
    result = {'my_balance':my_balance, 'avg_price':avg_price}
    return result

def f_my_coin(upbit):
    df = pd.DataFrame(upbit.get_balances())
    time.sleep(0.1)
    df.reset_index(drop=True, inplace=True)
    df['coin_name'] = df.unit_currency + '-' + df.currency
    df['buy_price'] = pd.to_numeric(df.balance, errors='coerce') * pd.to_numeric(df.avg_buy_price, errors='coerce')
    df = df[df.buy_price >= 5000]
    df.reset_index(drop=True, inplace=True)
    return df

def load_df(directory, name):
    ori_df = []
    json_file = directory + '/' + name
    try:
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')
    except:
        pass
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
                    time.sleep(0.5)
                    uuids.remove(uuid)
                except:
                    pass
        print("모든 예약 취소")
        json_file = directory + '/' + name
        df = pd.DataFrame(uuids)
        df = df.dropna(axis=0)
        df.to_json(json_file, orient='table')
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
        try:
            ori_df = pd.read_json(json_file, orient='table')
            df = pd.concat([ori_df, df], axis=0)
            df.reset_index(drop=True, inplace=True)
        except:
            df = df
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
    x = list()
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
    a = df.loc[:,'close'] - df.loc[:,'open']
    b = np.sum(a[a >= 0])
    c = abs(np.sum(a[a < 0]))
    rsi = round(b / (c + b) * 100, 2)
    return rsi
def f_bol(df):
    df = df[0:len(df)-1]
    bol_median = np.mean(df['close'])
    bol_std = np.std(df['close'])
    bol_higher = round(bol_median + 2 * bol_std,0)
    bol_lower = round(bol_median - 2 * bol_std,0)
    env_higher = round(bol_median * 1.03 + 2 * bol_std,0)
    env_lower = round(bol_median * 0.97 - 2 * bol_std,0)
    return bol_higher, bol_lower, env_higher, env_lower
def f_macd(df):
    ema12 = f_ema(df, 12)[-18:]
    ema26 = f_ema(df, 26)[-18:]
    macd = [i - j for i,j in zip(ema12, ema26)]
    signal = f_macd_signal(macd, 9)
    return signal
def f_ema(df, length):
    sdf = df[0:length]
    ema = round(np.mean(df.close[0:length]),0)
    n = np.count_nonzero(df.close.to_list())
    sdf = df[length:n-1]
    res = [ema]
    for i in range(np.count_nonzero(sdf.close)-1):
        ema = round(sdf.close[i+1]*2/(length+1) + ema*(1-2/(length+1)),2)
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
def generate_rate(coin_name, intervals, lines):
    res = []
    idx = 0
    for interval in intervals:
        idx += 1
        #interval = intervals[0]
        #coin_name = "KRW-POWR"
        #interval = "minute1"
        #lines = 10
        df = pyupbit.get_ohlcv(coin_name, interval = interval, count = 50)
        time.sleep(0.1)

        rsi_df = df.iloc[-lines:,:]
        rsi = f_rsi(rsi_df)
        bol_higher, bol_lower, env_higher, env_lower = f_bol(rsi_df)
        macd = f_macd(df)
        box = box_information(df)
        bdf = box[-7:].reset_index(drop=True)
        check_macd = bool(macd[-1]>0)
        bdf['macd'] = macd[-7:]

        coef_rate_vol  = f_reg_coef(bdf, 'rate_volume')  # 특정값보다 크면 좋음

        coef_macd = f_reg_coef(bdf,'macd')  # 증가 O
        coef_open = f_reg_coef(bdf, 'open')  # 증가 O
        coef_mid  = f_reg_coef(bdf, 'length_mid') # 0에 가까울수록 좋음 정체의미
        #거래량 많고, red 유형
        blue_bar = np.sum(bdf[bdf.color=='blue'].length_mid)
        red_bar = np.sum(bdf[bdf.color=='red'].length_mid)
        rate_red_bar = round(red_bar / (red_bar + blue_bar),3)
        vol_up = [bool(i == 'red') & bool(j > 2) for i, j in zip(bdf.color, bdf.rate_volume)]
        idx = intervals.index(interval)
        result = {'idx':idx,'coin_name': coin_name, 'rate_open':coef_open, 'interval': interval, 'volume_up': np.sum(vol_up),
                  'rate_red_bar':rate_red_bar, 'coef_macd':coef_macd, 'coef_mid':coef_mid, 'bol_higher':bol_higher, 'bol_lower':bol_lower, 'env_higher':env_higher,
                  'env_lower':env_lower, 'rsi':rsi}
        res.append(result)
    df = pd.DataFrame(res)
    return df

def coin_trade(upbit, ticekrs, investment, intervals, cutoff, benefit, lines, sec):
    th1 = Process(target=execute_buy_schedule, args=(upbit, ticekrs, intervals, investment, lines, sec))
    th2 = Process(target=execute_sell_schedule, args=(upbit, intervals, cutoff, benefit, lines))
    #th3 = Process(target=execute_search_schedule, args=(upbit, intervals, lines))
    result = Queue()
    th1.start()
    th2.start()
    th1.join()
    th2.join()

# input 1번 불러오면 되는 것들
if __name__ == '__main__':
    access_key = '5RsZuqMZ6T0tfyjNbIsNlKQ8LI4IVwLaYMBXiaa2'  # ''
    secret_key = 'zPKA1zJwymHMvUSQ2SqYWDgkxNgVfG7Z5jiNLcaJ'  # ''
    upbit = pyupbit.Upbit(access_key, secret_key)
    # intervals = ["day", "minute240", "minute60", "minute30", "minute15", 'minute10', 'minute5']
    intervals = ["minute5", "minute15", "minute30", "minute60", "minute240"]
    #intervals = ["minute10", "minute30", "minute60", "minute240"]
    #intervals = ["minute1", "minute60", "minute240", "day"]
    #intervals = ["minute1", "minute3", "minute5", "minute"]
    investment = 500000
    cutoff = 0.005
    benefit = 0.01
    lines = 20
    sec = 60  #
    ticekrs = []
    coin_trade(upbit, ticekrs, investment, intervals, cutoff, benefit, lines, sec)
