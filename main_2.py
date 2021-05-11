import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
from sklearn import linear_model
import os
from multiprocessing import Process, Queue
pd.set_option('display.max_columns', 30)

def execute_search_schedule(search_intervals, lines):
    try:
        os.remove('buy_selection_coin/buy_selection_coin.json')
        os.remove('sell_selection_coin/sell_selection_coin.json')
    except:
        pass
    while True:
        tickers = pyupbit.get_tickers(fiat="KRW")
        buy_res = []
        sell_res = []
        #search_intervals = ["minute120", "minute240"]
        for coin_name in tickers:
            #coin_name = tickers[0]
            try:
                buy_check, sell_check = auto_search(coin_name, search_intervals, lines)
                if buy_check:
                    buy_res.append(coin_name)
                if sell_check:
                    sell_res.append(coin_name)
            except:
                time.sleep(0.1)
                print("failed : search_coin")
        data_list_buy = pd.DataFrame(buy_res, columns=['coin_name']).reset_index(drop=True)
        data_list_sell = pd.DataFrame(sell_res, columns=['coin_name']).reset_index(drop=True)
        if len(data_list_buy) >= 1:
            merge_df(data_list_buy, 'buy_selection_coin', 'buy_selection_coin.json')
            print(str(datetime.now()) + "    구매 후보로 선정된 코인: " + ', '.join(data_list_buy.coin_name))
        if len(data_list_sell) >= 1:
            merge_df(data_list_sell, 'sell_selection_coin', 'sell_selection_coin.json')
            print(str(datetime.now()) + "    판매 후보로 선정된 코인: " + ', '.join(data_list_sell.coin_name))
        time.sleep(60*10*6)
def execute_buy_schedule(upbit, tickers, investment, lines):
    while True:
        reservation_cancel(upbit, 'buy_list', 'buy_list.json')
        money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
        while bool(money > investment):
            try:
                money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
            except:
                money = 0
            time.sleep(0.1)
            if money < max(5000, investment):
                pass
            else:
                if len(tickers) == 0:
                    load = []
                    while len(load) == 0:
                        load = load_df('buy_selection_coin', 'buy_selection_coin.json')
                    tickers = list(load.coin_name)
                #tickers = pyupbit.get_tickers("KRW")
                for coin_name in tickers:
                    #coin_name = tickers[0]
                    try:
                        auto_buy(upbit, coin_name, investment, lines)
                    except:
                        pass
                time.sleep(10)
def execute_sell_schedule(upbit, lines):
    while True:
        st = time.time()
        diff = 0
        buy_df = []
        sell_df = []
        reservation_cancel(upbit, 'sell_list', 'sell_list.json')
        while diff < 60:
            my = f_my_coin(upbit)
            time.sleep(1)
            tickers = (my.coin_name).to_list()
            sell_coin_list = []
            load = load_df('sell_selection_coin', 'sell_selection_coin.json')
            if len(load)>0:
                sell_tickers = list(load.coin_name)
                for coin_name in tickers:
                    if (coin_name in sell_tickers):
                        sell_coin_list.append(coin_name)
            if len(sell_coin_list) == 0:
                pass
            else:
                for coin_name in sell_coin_list:
                    try:
                        auto_sell(upbit, coin_name, lines)
                    except:
                        pass
                time.sleep(10)
def f_color_confirm(bdf):
    umbong = ['longbody_umbong','pole_umbong','shortbody_umbong', 'stone_cross', 'lower_tail_pole_umbong', 'upper_tail_pole_umbong']
    yangbong = ['longbody_yangbong','pole_yangbong','shortbody_yangbong', 'upper_tail_pole_yangbong', 'lower_tail_pole_yangbong']
    ls = bdf[-3:-1].chart_name.reset_index(drop=True)
    if bool(ls[0] in umbong) & bool(ls[1] in umbong):
        color_confirm_buy = True
    elif bool(ls[0] in yangbong) & bool(ls[1] in yangbong):
        color_confirm_sell = True
    return color_confirm_buy, color_confirm_sell

def f_coef_macd_confirm(bdf):
    bdf = bdf[-5:]
    coef_oscillator = f_reg_coef(bdf, 'oscillator')
    coef_open = f_reg_coef(bdf, 'open')
    return coef_oscillator, coef_open

def auto_search(coin_name, search_intervals, lines):
    # coin_name = 'KRW-TON'
    # interval = search_intervals[0]
    buy_check, sell_check = False, False
    for interval in search_intervals:
        dic = generate_rate_minute(coin_name, interval, lines)
        cv, interest, red_count , vr, up_rate, benefit, rsi = dic_valication(dic)
        price = pyupbit.get_current_price(coin_name)
        #print(interval+"/"+coin_name +"/"+dic.oscillator[8]+"/"+dic.bol_lower[8]+"/"+dic.env_lower[8])
        if (dic.oscillator[8] < 0) & ((dic.bol_lower[8] < price) | (dic.env_lower[8] < price)):
            buy_check = True
        if (dic.oscillator[8] > 0) & ((dic.bol_higher[8] > price) | (dic.env_higher[8] > price)):
            sell_check = True
    return buy_check, sell_check

def volume_value(dic):
    interest = sum(dic.rate_volume)
    red_count = sum(dic.color=='red')
    return interest, red_count
def color_value(dic):
    z = [(dic.close[len(dic)-1]-i) for i in dic.close]
    red_value = 0
    blue_value = 0
    for i in range(len(z)):
        if z[i] >= 0 :
            red_value += z[i]
        else:
            blue_value += z[i]
    value = (red_value + blue_value) * 0.8
    return value
def dic_valication(dic):
    dic = dic[-5:].reset_index(drop=True)
    cv = - color_value(dic)   # cv만큼 오를것이다.
    interest, red_count = volume_value(dic)    # 일치도 : red = 거래량상승 / blue = 거래량하락
    vr = sum(dic.rate_volume) # 거래량이 vr배 만큼 이루어지고 있다.
    up_rate = np.mean(dic.coef_open) # 시작가 상승기울기
    rsi = np.mean(dic.rsi)
    coef_oscillator = f_reg_coef(dic, 'oscillator')
    coef_mid = f_reg_coef(dic, 'length_mid')
    benefit = cv/dic.close[len(dic)-1]
    return cv, interest, red_count , vr, up_rate, benefit, rsi


def buy_check(dic, price):
    type = []
    idx = len(dic)-1
    if (dic.rsi[idx] < 30) & (dic.coef_oscillator[idx] > 0):
        type = '구매1'  # 패턴1. rsi < 30 & 상승
    elif (dic.bol_lower[idx] < price) & (dic.color[idx-2] == 'blue') & (dic.color[idx-1] == 'blue') & (
            dic.coef_oscillator[idx] > 0):
        type = '구매2'  # 패턴2. bol_lower & blue blue & 상승
    elif bool(np.sum(dic.color == 'blue') >= 4) & bool(
            dic.chart_name[idx] in ['stone_cross', 'lower_tail_pole_umbong', 'dragonfly_cross',
                                   'upper_tail_pole_yangong']):
        type = '구매3'
    elif bool(dic.chart_name[idx] in ['pole_yangbong', 'longbody_yangbong']):
        type = '구매4'
    elif bool(np.sum(dic.color == 'blue') >= 4) & bool(
            dic.chart_name[idx] in ['pole_umbong', 'doji_cross', 'spinning_tops']):
        type = '구매5'
    return type

def auto_buy(upbit, coin_name, investment, lines):
    #coin_name = 'KRW-LTC'
    #df = generate_rate(coin_name, intervals, lines)
    dic = generate_rate_minute(coin_name, "minute240", lines)
    cv, interest, red_count , vr, up_rate, benefit, rsi = dic_valication(dic)
    price = pyupbit.get_current_price(coin_name)
    if ((dic.bol_lower[4] >= price) | (dic.env_lower[4] >= price)) & (dic.coef_oscillator[4] < 0):
        dic = generate_rate_minute(coin_name, "minute60", lines)
        cv, interest, red_count , vr, up_rate, benefit, rsi = dic_valication(dic)
        if ((dic.bol_lower[4] >= price) | (dic.env_lower[4] >= price)) & (dic.coef_oscillator[4] < 0):
            dic = generate_rate_minute(coin_name, "minute30", lines)
            cv, interest, red_count , vr, up_rate, benefit, rsi = dic_valication(dic)
            if ((dic.bol_lower[4] >= price) | (dic.env_lower[4] >= price)) & (dic.coef_oscillator[4] < 0):
                dic = generate_rate_minute(coin_name, "minute10", lines)
                cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
                if ((dic.bol_lower[4] >= price) | (dic.env_lower[4] >= price)) & (dic.coef_oscillator[4] < 0):
                    dic = generate_rate_minute(coin_name, "minute1", lines)
                    cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
                    if ((dic.bol_lower[4] >= price) | (dic.env_lower[4] >= price)) & (dic.coef_oscillator[4] < 0):
                        print(coin_name+"/구매도달")
                        if (dic.bol_lower[4] >= price):
                            type = "bol"
                            upbit.buy_market_order(coin_name, investment)
                        elif (dic.env_lower[4] >= price):
                            type = "env"
                            price = coin_buy_price(coin_name)
                            investment = investment/3
                            count = investment / price
                            excute_buy(upbit, coin_name, price, count, investment)
                        else:
                            type = 'error'
                        print(str(datetime.now()) + "    ("+type+") 코인: " + coin_name +"    (예상) 수익: " + str(round(benefit,3)))
                    else:
                        print(str(datetime.now()) + "    (minute1 거절) 코인: " + coin_name)
                else:
                    print(str(datetime.now()) + "    (minute10 거절) 코인: " + coin_name)
            else:
                print(str(datetime.now()) + "    (minute30 거절) 코인: " + coin_name)
        else:
            print(str(datetime.now()) + "    (minute60 거절) 코인: " + coin_name)
    else:
        print(str(datetime.now()) + "    (minute240 거절) 코인: " + coin_name)

def check_criteria_buy(df,price):
    check = False
    type = '없음'
    if (df.color_buy_check[0]) & (df.oscillator[0] < 0):

        type = '구매1'
        check = True
    elif (df.oscillator[0] < 0) & (df.bol_lower[0] < price):  # 기본적인 조건 만족
        type = '구매2'
        check = True
    elif (df.rsi[0] < 30) & (df.oscillator[0] < -5):
        type ='구매3'
        check = True
    return type, check
def check_sell_case(dic):
    df = dic
    idx = len(df)
    result = False
    if bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong','upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-2] in ['longbody_yangbong','shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx-1] in ['spinning_tops','doji_cross']):
        result = True
    return result

def auto_sell(upbit, coin_name, lines):
    #coin_name = sell_coin_list[0]
    #coin_name = 'KRW-CBK'
    balance, avg_price = f_balance(upbit, coin_name)
    cur_price = pyupbit.get_current_price(coin_name)
    ratio = round((cur_price - avg_price) / avg_price, 3)

    price = cur_price
    dic = generate_rate_minute(coin_name, "minute240", lines)
    cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
    if (ratio >= -0.05):
        if (dic.bol_higher[4] <= price) | (dic.env_higher[4] <= price):
            dic = generate_rate_minute(coin_name, "minute60", lines)
            cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
            if ((dic.bol_higher[4] <= price) | (dic.env_higher[4] <= price)):
                dic = generate_rate_minute(coin_name, "minute30", lines)
                cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
                if ((dic.bol_higher[4] <= price) | (dic.env_higher[4] <= price)) :
                    dic = generate_rate_minute(coin_name, "minute10", lines)
                    cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
                    if ((dic.bol_higher[4] <= price) | (dic.env_higher[4] <= price)):
                        dic = generate_rate_minute(coin_name, "minute1", lines)
                        cv, interest, red_count, vr, up_rate, benefit, rsi = dic_valication(dic)
                        if (ratio > 0.01) & ((dic.bol_higher[4] <= price) | (dic.env_higher[4] <= price)) :
                            print(coin_name + "/판매도달")
                            if (dic.bol_higher[4] <= price):
                                upbit.sell_market_order(coin_name, balance)
                            elif (dic.env_higher[4] <= price):
                                price = coin_sell_price(coin_name)
                                balance = balance * 0.5
                                execute_sell(upbit, balance, coin_name, price)
                            print(str(datetime.now()) + "    (매도) 코인: " + coin_name + "    (예상) 수익: " + str(round(ratio, 3)))

                        else:
                            print(str(datetime.now()) + "    (매도) (minute1 거절) 코인: " + coin_name)
                    else:
                        print(str(datetime.now()) + "    (매도) (minute10 거절) 코인: " + coin_name)
                else:
                    print(str(datetime.now()) + "    (매도) (minute30 거절) 코인: " + coin_name)
            else:
                print(str(datetime.now()) + "    (매도) (minute60 거절) 코인: " + coin_name)
        else:
            print(str(datetime.now()) + "    (매도) (minute240 거절) 코인: " + coin_name)
    elif (ratio >= 0.01) & (check_sell_case(dic)):
        print(coin_name + "/판매도달")
        type = 'check_sell'
        price = coin_sell_price(coin_name)
        upbit.sell_market_order(coin_name, balance)
        # execute_sell(upbit, balance, coin_name, price)
        print(str(datetime.now()) + "    (" + type + ") 코인: " + coin_name + "    (예상) 수익: " + str(round(ratio, 3)))
    else:
        upbit.sell_market_order(coin_name, balance)
        print(str(datetime.now()) + "    (손절) 코인: " + coin_name)

def excute_buy(upbit, coin_name, price, count, investment):
    price = round_price(price)
    if price * count <= 5000:
        pass
    else:
        try:
            res = upbit.buy_limit_order(coin_name, price, count)
        except:
            count = investment / price
            res = upbit.buy_limit_order(coin_name, price, count)
        time.sleep(0.1)
        if len(res) > 2:
            print("***구매 요청 정보***")
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

def interest_trade(coin_name):
    #coin_name = 'KRW-BTC'
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    ask = np.sum(df_orderbook['ask_size'])
    bid = np.sum(df_orderbook['bid_size'])
    interest = 'nomal'
    if ask > bid:
        interest = 'sell'
    elif ask < bid:
        interest = 'buy'
    return interest
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
    no =0
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
    balance = my_df.balance.astype(float)[0]
    avg_price = my_df.avg_buy_price.astype(float)[0]
    return balance, avg_price

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
    json_file = directory + '/' + name
    ori_df = []
    if os.path.exists(json_file):
        ori_df = pd.read_json(json_file, orient='table')
    return ori_df

def reservation_cancel(upbit, directory, name):
    #directory = 'sell_list'
    #directory = 'buy_list'
    #name = 'sell_list.json'
    #name = 'buy_list.json'
    a = []
    df = load_df(directory, name)
    if len(df) > 0:
        uuids = list(set(df.uuid))
        while len(uuids) > 0:
            for uuid in uuids:
                while (uuid in uuids):
                    try:
                        res = upbit.cancel_order(uuid)
                        uuids.remove(uuid)
                        time.sleep(0.5)
                    except:
                        pass
        print("모든 예약 취소")
        json_file = directory + '/' + name
        df = pd.DataFrame(uuids)
        os.remove(json_file)
    else:
        pass

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
        df.to_json(json_file, orient='table')
    else:
        os.remove(json_file)
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
    bol_higher = round(bol_median + 2 * bol_std,3)
    bol_lower = round(bol_median - 2 * bol_std,3)
    env_higher = round(bol_median * (1+len(df)/1000),3)
    env_lower = round(bol_median * (1-len(df)/1000),3)
    return bol_median, bol_higher, bol_lower, env_higher, env_lower
def f_macd(df):
    ema12 = f_ema(df, 12)[-18:]
    ema26 = f_ema(df, 26)[-18:]
    macd = [i - j for i, j in zip(ema12, ema26)][-9:]
    signal = f_macd_signal(macd, 9)[-9:]
    oscillator = [i - j for i, j in zip(macd, signal)]
    return macd, signal, oscillator
def f_ema(df, length):
    #df = df2
    #length = 12
    sdf = df[0:length].reset_index(drop=True)
    ema = round(np.mean(df.close[0:length]),0)
    n = np.count_nonzero(df.close.to_list())
    sdf = df[length:n-1]
    res = [ema]
    ls = list(sdf.close)
    for i in range(np.count_nonzero(ls)-1):
        ema = round(ls[i+1]*2/(length+1) + ema*(1-2/(length+1)),2)
        res = res + [ema]
    return res
def f_macd_signal(macd, length):
    length  = 9
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

def generate_rate_minute(coin_name, interval, lines):
    #interval = 'minute3'
    #coin_name = 'KRW-BTT'
    #lines = 20
    df = pyupbit.get_ohlcv(coin_name, interval = interval, count = 50)
    time.sleep(0.1)
    rsi_df = df.iloc[-lines:,:]
    rsi = f_rsi(rsi_df)
    bol_median, bol_higher, bol_lower, env_higher, env_lower = f_bol(rsi_df)
    macd, signal, oscillator = f_macd(df)  ################# check1
    box = box_information(df)
    bdf = box[-9:].reset_index(drop=True)
    bdf["macd"] = macd
    bdf["signal"] = signal
    bdf["oscillator"] = oscillator
    coef_oscillator, coef_open = f_coef_macd_confirm(bdf)  ### check3
    bdf["coef_oscillator"] = coef_oscillator
    bdf["coef_open"] = coef_open
    bdf["rsi"] = rsi
    bdf["bol_lower"] = bol_lower
    bdf["bol_median"] = bol_median
    bdf["bol_higher"] = bol_higher
    bdf["env_lower"] = env_lower
    bdf["env_higher"] = env_higher
    #거래량 많고, red 유형
    result = bdf
    return result

def coin_trade(upbit, tickers, investment, search_intervals, lines):
    th3 = Process(target=execute_search_schedule, args=(search_intervals, lines))
    th1 = Process(target=execute_buy_schedule, args=(upbit, tickers, investment, lines))
    th2 = Process(target=execute_sell_schedule, args=(upbit, lines))
    #th3 = Process(target=execute_search_schedule, args=(upbit, intervals, lines))
    result = Queue()
    th3.start()
    th1.start()
    th2.start()
    th3.join()
    th1.join()
    th2.join()

# input 1번 불러오면 되는 것들
if __name__ == '__main__':
    access_key = '5RsZuqMZ6T0tfyjNbIsNlKQ8LI4IVwLaYMBXiaa2'  # ''
    secret_key = 'zPKA1zJwymHMvUSQ2SqYWDgkxNgVfG7Z5jiNLcaJ'  # ''
    upbit = pyupbit.Upbit(access_key, secret_key)
    search_intervals = ["day"]
    investment = 100000
    lines = 20
    tickers = []
    coin_trade(upbit, tickers, investment, search_intervals, lines)
