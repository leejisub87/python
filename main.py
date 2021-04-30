import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
import os
from multiprocessing import Process, Queue
pd.set_option('display.max_columns', 20)

#################################### data generate
def get_hoga_price(coin_name):
    orderbook = []
    result = []
    while not orderbook:
        orderbook = pyupbit.get_orderbook(coin_name)
        time.sleep(1)
        try:
            df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
            ask_price = df_orderbook.ask_price
            bid_price = df_orderbook.bid_price
            result = {'ask_price': ask_price, 'bid_price': bid_price}
        except:
            pass
    return result
def execute_sell(upbit, type, balance, coin_name, avg_price, current_price, price):
    if price * balance > 5000:
        count = balance
        weight = 1
        count = count * weight
        r = [3,2,1,0,-1,-2,-3,-4,-5,-6,-7]
        try:
            for i in r:
                res = upbit.sell_limit_order(coin_name, round(price,i), count)
                time.sleep(0.5)
                if len(res) > 2:
                    ratio = (price - avg_price) / avg_price
                    print("매도 요청("+type+"): " + coin_name + ", 현재가: " + str(current_price) + ", 손익률: " + str(
                        round(ratio * 100, 2)) + "%, 매도 호가(" + type + "): " + str(price))
                    print(res.get('uuid')[0])
                    save_uuid(res.get('uuid')[0])
                    time.sleep(0.5)
                    break
        except:
            pass
    else:
        print("판매 실패: "+coin_name +"은 판매 최소금액이 부족")

def execute_sell_schedule(upbit, cutoff, benefit):
    time.sleep(30)
    while True:
        st = time.time()
        diff = 0
        buy_df = []
        sell_df = []
        reservation_cancel(upbit)
        while diff < 60:
            while len(sell_df) == 0:
                try:
                    sell_df = load_df()
                except:
                    pass

            now = datetime.now()
            #sell_df = sell_df[sell_df.start_date_bid <= now]
            sell_df = sell_df[sell_df.end_date_bid >= now]
            #sell_df = sell_df[sell_df.check_sell]
            sell_df.sort_values('price_changed_ratio', ascending=False, inplace=True)
            sell_df.reset_index(drop=True, inplace=True)
            my_coin = pd.DataFrame(upbit.get_balances())

            my_coin['coin_name'] = my_coin.unit_currency +'-'+my_coin.currency
            my_coin['buy_price'] = pd.to_numeric(my_coin.balance, errors='coerce') * pd.to_numeric(my_coin.avg_buy_price, errors='coerce')
            KRW = float(my_coin[0:1].balance)
            tot_investment = round(KRW + sum(my_coin['buy_price']),0)

            my_coin = my_coin[pd.to_numeric(my_coin.avg_buy_price) > 0]
            my_coin = my_coin[my_coin.buy_price > 5000]
            my_coin.sort_values('buy_price', ascending=False, inplace=True)
            my_coin.reset_index(drop=True, inplace=True)

            price_cutoff = np.mean(sell_df.price_changed_ratio)

            predict_tickers = list(sell_df.coin_name)  # target price
            my_tickers = list(my_coin.coin_name)

            up_count = np.sum(sell_df.price_changed_ratio > 0)

            if up_count == 0:
                up_rate = 0
            elif len(sell_df) == 0:
                up_rate = round(up_count / len(my_coin), 2)
            else:
                up_rate = round(up_count / len(sell_df), 2)
            time.sleep(0.5)

            print("***************************** ")
            print("********** 판매 정보 ********** ")
            print("총투자금액: " + str(tot_investment) + "원, 개수: " + str(len(sell_df)) + "개, 상승률: " + str(
                round(up_rate * 100, 2)) + "%, 입력수익률 : " + str(round(benefit * 100, 2)) + "%, 입력손절률: -" + str(
                round(cutoff * 100, 2)) + "%")

            for coin_name in my_tickers:
                coin_name = 'KRW-ETH'
                df = my_coin[my_coin.coin_name == coin_name].reset_index(drop=True)
                balance = float(df.balance[0])
                avg_price = float(df.avg_buy_price)
                current_price = pyupbit.get_current_price(coin_name)
                time.sleep(0.2)
                ratio = (current_price - avg_price) / avg_price
                hoga = get_hoga_price(coin_name)
                price = avg_price
                min_benefit = benefit
                weight = 1
                if coin_name in predict_tickers:
                    sdf = sell_df[sell_df.coin_name == coin_name].reset_index(drop=True)
                    weight = sdf.price_changed_ratio[0]
                    sell_price = np.mean(sdf.sell_price_min) + 1.96 * np.std(sdf.sell_price_min)/np.sqrt(len(sdf.sell_price_min))
                    price = max(avg_price, sell_price)
                    min_benefit = min_benefit + min_benefit * sdf.price_changed_ratio[0]
                min_benefit = max(min_benefit, 0.01)
                price = price * (1 + min_benefit)

                min_cutoff = cutoff - price_cutoff

                if ratio < - min_cutoff:
                    type = '손절'
                    price = current_price
                    execute_sell(upbit, type, balance, coin_name, avg_price, current_price, price)

                elif ratio > benefit:
                    type = '익절'
                    if np.sum(hoga.get('ask_price') > price) >= 1:
                        price = min(hoga.get('ask_price')[hoga.get('ask_price') > price])
                    elif np.sum(hoga.get('bid_price') > price) >= 1:
                        price = hoga.get('bid_price')[0]
                    else:
                        price = current_price
                    execute_sell(upbit, type, balance, coin_name, avg_price, current_price, price)

                else:
                    print("판매 대기: "+coin_name+" 현재가: " + str(current_price) +"("+str(round(ratio*100,2))+"%), 목표률: "+str(round(benefit*100,2)) +"%, 손익률: " + str(round(ratio * 100, 2)))

def excute_buy(upbit, df, coin_name, investment):
    if len(df) == 0:
        print("데이터가 없습니다.")
    else:
        for i in range(len(df)):
            #i =1
            df = df[i : i + 1].reset_index(drop=True)
            status = 'wait'
            temp_price = df.buy_price_max[0]
            money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
            time.sleep(1)
            # 구매 가격 결정하기
            if money < max(5000, investment):
                print("주문금액 부족")
            else:
                last_investment = min(investment, money)
                hoga = get_hoga_price(coin_name)
                criteria = max(hoga.get('ask_price')) > temp_price > min(hoga.get('bid_price'))
                if not criteria:
                    print("************************************")#print(coin_name + "은 구매 계획에 도달하지 않았다.")
                else:
                    price_set = list(hoga.get('bid_price')[0:3])
                    print(coin_name+" 구매 호가: "+ str(price_set))

                    for price in price_set:
                        #price = price_set[0]
                        weight =  (price - last_investment)/temp_price # -값이 작아질수록 좋다. 갭이 커질수록
                        investment = round(investment * (1-weight), -2)
                        count = investment / price
                        print("investment money: " + str(count * price))
                        try:
                            res = upbit.buy_limit_order(coin_name, price, count)
                            time.sleep(1)
                            if len(res) > 2:
                                print("구매 성공: " + coin_name + "/ " + str(price))
                                save_uuid(res.get('uuid'))
                            else:
                                pass
                        except:
                            pass
def execute_buy_schedule(upbit, investment):
    time.sleep(30)
    while True:
        st = time.time()
        diff = 0
        buy_df = []
        reservation_cancel(upbit)
        while diff < 60:
            money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
            time.sleep(0.5)

            if money <= max(5000, investment):
                print("주문금액 부족")
            else:
                while len(buy_df) == 0:
                    try:
                        buy_df = load_df()
                    except:
                        pass
                now = datetime.now()
                buy_df = buy_df[buy_df.check_buy]
                buy_df = buy_df[buy_df.start_date_ask <= now]
                buy_df = buy_df[buy_df.end_date_ask >= now]
               # buy_df = buy_df[buy_df.price_changed_ratio > 0]
                buy_df.reset_index(drop = True, inplace =True)
                buy_df.sort_values('price_changed_ratio', inplace=True, ascending = False)
                first_res = []
                tickers = list(set(buy_df.coin_name)) # 구매신청
                if len(buy_df) > 0:
                    for coin_name in tickers:
                        df = buy_df[buy_df.coin_name == coin_name]
                        df.reset_index(drop=True, inplace=True)
                        try:
                            excute_buy(upbit, df, coin_name, investment)
                        except:
                            pass
                        time.sleep(0.1)
            et = time.time()
            diff = et - st

##################################### data generate
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

def coin_predict(default_df,benefit):
    df = default_df[-5:]
    df.reset_index(inplace=True, drop = True)
    default = df[-1:].reset_index()
    influence_m = np.sum(np.mean(df.length_mid) < df.length_mid)
    influence_v = np.sqrt(pow(np.sum(np.mean(df.volume) < df.volume),2) * pow(influence_m,2))
    min_buy_price = min(default.open[0], default.close[0]) - influence_v * (default.avg_length_low[0] - 1.96 * np.sqrt(default.std_length_low[0]))
    max_buy_price = min(default.open[0], default.close[0]) - influence_v * (default.avg_length_low[0] + 1.96 * np.sqrt(default.std_length_low[0]))
    #max_sell_price = max(default.open[0], default.close[0]) - influence_v * (default.avg_length_high[0] + 1.96 * np.sqrt(default.std_length_high[0]))

    min_sell_price = max(default.open[0], default.close[0]) - influence_v * (default.avg_length_high[0] - 1.96 * np.sqrt(default.std_length_high[0]))
    min_sell_price = max(min_sell_price, max_buy_price * (1+benefit))
    check_buy = check_buy_case(df)
    check_sell = check_sell_case(df)
    # 상태
    result = {'check_buy':check_buy,'check_sell': check_sell,'buy_price_max': max_buy_price, 'sell_price_min': min_sell_price}
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

def box_vervus(default_res,benefit):
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
    result = coin_predict(default_df, benefit)
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
    if len(df)>0:
        df_1 = df[-1:].reset_index(drop=True)
        changed_ratio = (df_1.close[0] - df_1.open[0]) / df_1.open[0]
        result = changed_ratio
    return result
def coin_information(coin_name, interval, benefit):
    price_currency = pyupbit.get_current_price(coin_name)
    df = pyupbit.get_ohlcv(coin_name, interval=interval, count=10)
    time.sleep(1)
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
def coin_information_interval(tickers, interval,benefit):
    li = []
    for coin_name in tickers:
        #coin_name = tickers[1]
        res = coin_information(coin_name, interval, benefit)
        li.append(res)
    df = pd.DataFrame(li)
    result = df
    return result
def coin_validation(upbit, interval, benefit):
    tickers = pyupbit.get_tickers(fiat="KRW")
    st = time.time()
    print("분석 시작 : "+interval)
    df = coin_information_interval(tickers, interval, benefit)
    result = merge_df(df)
    et = time.time()
    diff = round(et-st,2)
    print("분석 종료 : " + str(diff)+"초")
    print("selection coin count: "+str(len(set(result.coin_name)))+", avg_changed_price: "+str(round(np.mean(result.price_changed_ratio)*100,5))+"%")
    return result


########################## data
def save_uuid(uuid):
    #uuid = res
    # uuid -> dataframe, 기존의 uuid를 불러온다. uuid를 저장한다
    df = pd.DataFrame(uuid, columns=['uuid']).reset_index(drop=True)
    #df = df.dropna(axis=0)

    name = 'reservation_list.json'
    directory = 'reservation'
    json_file = directory+'/'+name
    # 폴더생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        # 데이터 읽고 머지
    file_list = os.listdir(directory)
    mdf = df
    if len(file_list) >= 1:
        ori_df = pd.read_json(json_file, orient='table')
        mdf = pd.concat([ori_df.reset_index(drop=True), mdf.reset_index(drop=True)], axis=0)
        mdf.reset_index(drop=True, inplace=True)

    while len(file_list) == 0:
        os.remove(json_file)
        file_list = os.listdir(directory)

    mdf.to_json(json_file, orient='table')
    return mdf
def load_uuid():
    # uuid -> dataframe, 기존의 uuid를 불러온다. uuid를 저장한다
    directory = 'reservation'
    ori_df= []
    if os.path.exists(directory) :
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')
    else:
        print("예약 내역이 없습니다.")
    return ori_df
def load_df():
    name = 'schedule_list.json'
    directory = 'schedule'
    json_file = directory + '/' + name
    ori_df = pd.read_json(json_file, orient='table')
    ori_df = pd.DataFrame(ori_df).reset_index(drop=True)
    return ori_df

def merge_df(df):
    mdf = df
    name = 'schedule_list.json'
    directory = 'schedule'
    json_file = directory + '/' + name
# 폴더생성
    while not os.path.exists(directory):
        os.makedirs(directory)

    file_list = os.listdir(directory)
    if len(file_list) >= 1:
        ori_df = pd.read_json(json_file, orient='table')
        mdf = pd.concat([ori_df.reset_index(drop=True), mdf.reset_index(drop=True)], axis=0)
        mdf.reset_index(drop = True, inplace = True)

    now = datetime.now()
    mdf = mdf[now <= mdf.end_date_bid]
    mdf.reset_index(drop=True, inplace=True)
    while len(file_list) >= 1 :
        os.remove(json_file)
        file_list = os.listdir(directory)
    mdf.to_json(json_file, orient='table')
    return mdf


######################### cancel
def convert(set):
    return [*set, ]

def reservation_cancel(upbit):
    df = []
    try:
        df = load_uuid()
    except:
        pass
    if len(df) > 0:
        uuids = convert(set(df.uuid))
        while len(uuids)>0:
            for uuid in uuids:
                status = 'wait'
                res = []
                while len(res)==0:
                    try:
                        res = upbit.cancel_order(uuid)
                        time.sleep(1)
                        uuids.remove(uuid)
                    except:
                        pass
        print("모든 예약 취소")
        save_uuid(uuids)

def multi_coin_validation(upbit, benefit):
    #interval = intervals[0]
    intervals = ["day", "minute240", "minute60", "minute30", "minute15", 'minute10', 'minute5']
    tt = list()
    at = time.time()
    while True:
        s1 = time.time()
        diff1 = 0
        while diff1 < 60*60*24:
            interval = intervals[0]
            coin_validation(upbit, interval, benefit)
            print("completed: " + interval)
            et = time.time()
            diff1 = et - s1
            s2 = time.time()
            diff2 = 0
            while diff2 < 60 * 60 * 4:
                interval = intervals[1]
                coin_validation(upbit, interval, benefit)
                print("completed: " + interval)
                et = time.time()
                diff2 = et - s2
                s3 = time.time()
                diff3 = 0
                while diff3 < 60 * 60 * 1:
                    interval = intervals[2]
                    coin_validation(upbit, interval, benefit)
                    print("completed: " + interval)
                    et = time.time()
                    diff3 = et - s3
                    s4 = time.time()
                    diff4 = 0
                    while diff4 < 60 * 30:
                        interval = intervals[3]
                        coin_validation(upbit, interval, benefit)
                        print("completed: " + interval)
                        et = time.time()
                        diff4 = et - s4
                        s5 = time.time()
                        diff5 = 0
                        while diff5 < 60 * 15:
                            interval = intervals[4]
                            coin_validation(upbit, interval, benefit)
                            print("completed: " + interval)
                            et = time.time()
                            diff5 = et - s5
                            s6 = time.time()
                            diff6 = 0
                            while diff6 < 60 * 10:
                                interval = intervals[5]
                                coin_validation(upbit, interval, benefit)
                                print("completed: " + interval)
                                et = time.time()
                                diff6 = et - s6
                                s7 = time.time()
                                diff7 = 0
                                while diff7 < 60 * 5:
                                    interval = intervals[6]
                                    coin_validation(upbit, interval, benefit)
                                    print("completed: " + interval)
                                    et = time.time()
                                    diff7 = et - s7
        bt = time.time()
        tt = bt - at
        print("total process time: " + str(tt) + "(s)")

def coin_trade(upbit, investment, cutoff, benefit):
    th1 = Process(target=multi_coin_validation, args=(upbit, benefit))
    th2 = Process(target=execute_buy_schedule, args=(upbit, investment))
    th3 = Process(target=execute_sell_schedule, args=(upbit, cutoff, benefit))
    result = Queue()
    th1.start()
    th2.start()
    th3.start()
    th1.join()
    th2.join()
    th3.join()


# input 1번 불러오면 되는 것들
if __name__ == '__main__':
    access_key = 'DXqzxyCQkqL9DHzQEyt8pK5izZXU03Dy2QX2jAhV'  # ''
    secret_key = 'x6ubxLyUVw03W3Lx5bdvAxBGWI7MOMJjblYyjFNo'  # ''
    upbit = pyupbit.Upbit(access_key, secret_key)
    investment = 5000
    cutoff = 0.009
    benefit = 0.019
    coin_trade(upbit, investment, cutoff, benefit)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
