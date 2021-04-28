import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
import os
pd.set_option('display.max_columns', 20)


#################################### data generate
def get_hoga_price(coin_name):
    orderbook = []
    result = []
    while not orderbook:
        orderbook = pyupbit.get_orderbook(coin_name)
        try:
            df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
            ask_price = df_orderbook.ask_price
            bid_price = df_orderbook.bid_price
            result = {'ask_price': ask_price, 'bid_price': bid_price}
        except:
            pass
    return result
def execute_sell_schedule(upbit, sell_df, cutoff, benefit):
    # mdf = sell_df
    # sell_df = new_df
    my_coin = pd.DataFrame(upbit.get_balances())
    my_coin['coin_name'] = my_coin.unit_currency +'-'+my_coin.currency
    my_coin['buy_price'] = pd.to_numeric(my_coin.balance, errors='coerce') * pd.to_numeric(my_coin.avg_buy_price, errors='coerce')
    KRW = float(my_coin[0:1].balance)
    my_coin = my_coin[pd.to_numeric(my_coin.avg_buy_price) > 0]
    predict_tickers = list(sell_df.coin_name) # target price
    my_tickers = list(my_coin.coin_name)

    print("잔액: "+ str(KRW))
    for coin_name in my_tickers:
        df = my_coin[my_coin.coin_name == coin_name].reset_index(drop=True)
        current_price = pyupbit.get_current_price(coin_name)
        balance = float(df.balance[0])
        avg_price = float(df.avg_buy_price)
        ratio = (current_price - avg_price) / avg_price
        hoga = get_hoga_price(coin_name)
        min_price = avg_price * benefit

        if ratio < -cutoff:
            type = '손절'
            price_set = current_price
        else:
            type = '익절'
            ho_ask = hoga.get('ask_price')[hoga.get('ask_price') > min_price]
            ho_bid = hoga.get('bid_price')[hoga.get('bid_price') > min_price]
            ho_list = pd.concat([(ho_ask),(ho_bid)], axis = 0).reset_index(drop=True)
            if bool(len(ho_list) >= 3):
                price_set = ho_list[-3:]
            else:
                price_set = ho_list[-2:]
            price_set = list(price_set)
        print("매도 정보: "+coin_name)
        print("매도 호가("+type+"): " + str(price_set) +", 현재가: "+str(current_price)+", 구매가: "+str(avg_price)+", 손익률: "+str(round(ratio*100,2))+"%")
        for price in price_set:
            # price = price_set[0]
            count = investment / price
            try:
                res = upbit.sell_limit_order(coin_name, price, count)
                time.sleep(0.1)
                if len(res) > 2:
                    print("매도 성공("+type+"): " + coin_name + "/ " + str(price))
                    save_uuid(res)
                else:
                    print("매도 실패("+type+"): " + coin_name + "/ " + str(price))
            except:
                print('매도 error: ' + coin_name)

    # 판매 가격 결정하기...
def excute_buy_df(upbit, df, coin_name, investment):
    for i in range(len(df)):
        #i =0
        df = df[i : i + 1]
        status = 'wait'
        temp_price = df.buy_price_max[0]
        money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
        # 구매 가격 결정하기
        if money < max(5000, investment):
            print("주문금액 부족")
        else:
            last_investment = min(investment, money)
            hoga = get_hoga_price(coin_name)
            criteria = max(hoga.get('ask_price')) > temp_price > min(hoga.get('bid_price'))
            if not criteria:
                print(coin_name + "은 구매 계획에 도달하지 않았다.")
            else:
                print(coin_name + "은 구매 계획에 도달했다.")
                ho = list(hoga.get('bid_price')[hoga.get('bid_price') <= temp_price])
                if len(ho) >= 3:
                    price_set = ho[-3:]
                else:
                    price_set = ho[-2:]
                print("구매 호가: "+ str(price_set))

                for price in price_set:
                    #price = price_set[0]
                    count = investment / price
                    print(count * price)
                    try:
                        res = upbit.buy_limit_order(coin_name, price, count)
                        time.sleep(0.1)
                        if len(res) > 2:
                            print("구매 성공: " + coin_name + "/ " + str(price))
                            save_uuid(res)
                        else:
                            print("구매 실패: " + coin_name + "/ " + str(price))
                    except:
                        print('매수 error: ' + coin_name)

def execute_buy_schedule(upbit, buy_df, investment):
    money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
    if money < max(5000, investment) :
        print("주문금액 부족")
    else:
        # mdf = buy_df
        buy_df.reset_index(drop=True, inplace=True)
        first_res = []
        tickers = list(set(buy_df.coin_name)) # 구매신청
        for coin_name in tickers:
            df = buy_df[buy_df.coin_name == coin_name]
            df.reset_index(drop=True, inplace=True)
            excute_buy_df(upbit, df, coin_name, investment)

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
    a = criteria_updown(df[-2:])
    b = criteria_updown(df[-3:-1])
    idx = len(df)
    result = False
    if bool(np.sum(df.color == 'red')>=3) & bool(df.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong','upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'red')>=3) & bool(df.chart_name[idx-2] in ['longbody_yangbong','shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'red')>=3) & bool(df.chart_name[idx-1] in ['spinning_tops','doji_cross']):
        result = True
    elif bool(b.get('volume_status') == 'up') & bool(b.get('open_status') == 'up') & bool(a.get('volume_status') == 'down') & bool(a.get('open_status') == 'down'):
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
    result = {'volume_status':volume_status, 'close_rate':close_rate, 'open_status':open_status}
    return result
def coin_predict(default_df,benefit):
    df = default_df[-5:]
    df.reset_index(inplace=True)
    default = df[-1:].reset_index()
    influence_m = np.sum(np.mean(df.length_mid) < df.length_mid)
    influence_v = np.sqrt(pow(np.sum(np.mean(df.volume)< df.volume),2) * pow(influence_m,2))
    #min_buy_price = min(default.open[0], default.close[0]) - influence_v * (default.avg_length_low[0] - 1.96 * np.sqrt(default.std_length_low[0]))
    max_buy_price = min(default.open[0], default.close[0]) - influence_v * (default.avg_length_low[0] + 1.96 * np.sqrt(default.std_length_low[0]))
    #max_sell_price = max(default.open[0], default.close[0]) - influence_v * (default.avg_length_high[0] + 1.96 * np.sqrt(default.std_length_high[0]))

    min_sell_price = max(default.open[0], default.close[0]) - influence_v * (default.avg_length_high[0] - 1.96 * np.sqrt(default.std_length_high[0]))
    min_sell_price = max(min_sell_price, max_buy_price * benefit)
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
    default_df = []
    df = default_res[-5:]
    for i in range(len(df)):
        a = default_res[i:i+1]
        b = default_res[i+1:i+2]
        result = []
        for col_name in list(a.columns):
            try:
                c = (b[col_name] - a[col_name])/a[col_name]
                result.append({'col_name':col_name, 'value':c[0]})
            except:
                pass
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
# def box_vervus(default_res):
#     default_df = []
#     for i in range(len(default_res)-1):
#         a = default_res[i:i+1]
#         b = default_res[i+1:i+2]
#         result = []
#         for col_name in list(a.columns):
#             try:
#                 c = (b[col_name] - a[col_name])/a[col_name]
#                 result.append({'col_name':col_name, 'value':c[0]})
#             except:
#                 pass
#         df_1 = pd.DataFrame(result)
#         columns_name = list('vs_'+df_1.col_name)
#         value = list(df_1.value)
#         b = b.reset_index(drop=True)
#         for j in range(len(value)):
#             b[columns_name[j]] = value[j]
#         if i == 0 :
#             default_df = b
#         else:
#             default_df = pd.concat([default_df, b], axis=0)
#     result = coin_predict(default_df)
#     return result
def box_information(df,benefit):
    default_res = []
    for i in range(len(df)):
        df_1 = df[i:i+1]
        if i == 0:
            default_res = box_create(df_1)
        else:
            next_res = box_create(df_1)
            default_res = pd.concat([default_res, next_res], axis=0)
    result = box_vervus(default_res, benefit)
    return result
def changed_ratio(df, price_currency):
    df_1 = df[-1:]
    changed_ratio = (df_1.close[0] - df_1.open[0]) / df_1.open[0]
    result = changed_ratio
    return result
def coin_information(coin_name, interval, benefit):
    price_currency = pyupbit.get_current_price(coin_name)
    df = pyupbit.get_ohlcv(coin_name, interval=interval, count=10)
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
def coin_validation(upbit, tickers, interval, benefit):
    st = time.time()
    print("분석 시작 : "+interval)
    df = coin_information_interval(tickers, interval, benefit)
    df[df.check_buy].reset_index(drop=True, inplace=True)
    res = merge_df(df)
    et = time.time()
    diff = round(et-st,2)
    result = res[res.check_buy].reset_index(drop=True)
    print("분석 종료 : " + str(diff)+"초")
    print("selection coin count: "+str(len(result.coin_name))+", avg_changed_price: "+str(round(np.mean(result.price_changed_ratio),5)*100)+"%")
    return result


########################## data
def save_uuid(uuid):
    #uuid = res
    # uuid -> dataframe, 기존의 uuid를 불러온다. uuid를 저장한다
    df = pd.DataFrame([uuid.get('uuid')], columns=['uuid']).reset_index(drop=True)
    #df = df.dropna(axis=0)

    name = 'reservation_list.json'
    directory = 'reservation'
    json_file = directory+'/'+name
    # 폴더생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        # 데이터 읽고 머지
    file_list = os.listdir(directory)
    merge_df = df
    if len(file_list) >= 1:
        ori_df = pd.read_json(json_file, orient='table')
        merge_df = pd.concat([ori_df.reset_index(drop=True), merge_df.reset_index(drop=True)], axis=0)
        merge_df.reset_index(drop=True, inplace=True)

    try:
        os.remove(json_file)
    except:
        print("not exsist file: schedule_list.json")

    merge_df.to_json(json_file, orient='table')
    return merge_df
def load_uuid():
    # uuid -> dataframe, 기존의 uuid를 불러온다. uuid를 저장한다
    directory = 'reservation'
    ori_df= []
    if not os.path.exists(directory) :
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')

    else:
        print("예약 내역이 없습니다.")

    return ori_df
def load_df():
    dt_now = datetime.now()
    now = dt_now.strftime('%Y%m%d')
    name = 'schedule_' + now + '.json'
    directory = 'schedule'
    json_file = directory + '/' + name
    name = os.listdir(directory)[-1]
    json_file = directory + '/' + name
    ori_df = pd.read_json(json_file, orient='table')
    ori_df = pd.DataFrame(ori_df)
    return ori_df
def merge_df(df):
    #df = new_df
    merge_df = df
    name = 'schedule_list.json'
    directory = 'schedule'
    json_file = directory + '/' + name
# 폴더생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("creation folder")

    file_list = os.listdir(directory)
    if len(file_list) >= 1:
        ori_df = pd.read_json(json_file, orient='table')
        merge_df = pd.concat([ori_df.reset_index(drop=True), merge_df.reset_index(drop=True)], axis=0)
        merge_df.reset_index(drop = True, inplace = True)
    now = datetime.now()
    merge_df = merge_df[now <= merge_df.end_date_bid]
    merge_df.reset_index(drop=True, inplace=True)
    try:
        os.remove(json_file)
    except:
        print("not exsist file: schedule_list.json")
    merge_df.to_json(json_file, orient='table')
    return df


######################### cancel
def convert(set):
    return [*set, ]
def reservation_cancel(upbit):
    df = load_uuid()
    if len(df) > 0:
        uuids = convert(set(df.uuid))
        while len(uuids)>0:
            for uuid in uuids:
                status = 'wait'
                try:
                    res = upbit.cancel_order(uuid)
                    time.sleep(0.1)
                    uuids.remove(uuid)
                except:
                    pass
        print("모든 예약 취소")
    else :
        print("예약 없음")

def coin_trade(upbit, interval, total_updown, investment, cutoff, benefit):
    intervals = ["month", "week", "day", "minute240", "minute60", "minute30", "minute10", 'minute5']
    tickers = pyupbit.get_tickers(fiat="KRW")
    print("현황 분석 시작.")
    #new_df = load_df()
    new_df = coin_validation(upbit, tickers, interval, benefit)
    print("현황 분석 완료.")

    price_benefit = np.mean(new_df.price_changed_ratio[new_df.price_changed_ratio>0])
    price_cutoff = np.mean(new_df.price_changed_ratio)
    count = np.sum(new_df.price_changed_ratio > 0)
    up_rate = count/len(new_df)
    print("selection coin information")
    print("개수: "+str(len(new_df))+"개, 상승률: "+str(round(up_rate*100,2))+"%, 예상 수익률 : "+str(round(price_benefit*100,2))+"%")
    buy_df = new_df
    print("buying schedule")
    execute_buy_schedule(upbit, buy_df, investment)

    sell_df = new_df
    cutoff = abs(max(price_cutoff, cutoff))
    print("손절률: "+ str(round(cutoff*100,2))+"%")

    print("selling schedule")
    execute_sell_schedule(upbit, sell_df, cutoff, price_benefit)

    reservation_cancel(upbit)

    result = price_benefit
    return result


# input 1번 불러오면 되는 것들
if __name__ == '__main__':
    intervals = ["month", "week", "day", "minute240", "minute60", "minute30", "minute10", 'minute5']
    access_key = 'DXqzxyCQkqL9DHzQEyt8pK5izZXU03Dy2QX2jAhV'
    secret_key = 'x6ubxLyUVw03W3Lx5bdvAxBGWI7MOMJjblYyjFNo'

    upbit = pyupbit.Upbit(access_key, secret_key)
    interval = intervals[7] # 0 ~ 7
    benefit = 0.01
    investment = 10000
    cutoff = 0.015
    total_updown = []
    while True:
        try:
            res = coin_trade(upbit, interval, total_updown, investment, cutoff, benefit)
            total_updown.append(res)
        except:
            pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
