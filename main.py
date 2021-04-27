import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
import os
pd.set_option('display.max_columns', 20)

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
def reservation_cancel(upbit):
    df = load_uuid()
    if len(df) > 0:
        uuids = list(set(df.uuid))
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
def execute_sell_schedule(upbit, sell_df, cutoff, benefit):
    # mdf = buy_df
    my_coin = pd.DataFrame(upbit.get_balances())
    my_coin['coin_name'] = my_coin.unit_currency +'-'+my_coin.currency
    my_coin['buy_price'] = pd.to_numeric(my_coin.balance, errors='coerce') * pd.to_numeric(my_coin.avg_buy_price, errors='coerce')
    KRW = float(my_coin[0:1].balance)

    my_coin = my_coin[pd.to_numeric(my_coin.avg_buy_price) > 0]
    ticker_predict = sell_df.coin_name
    ticker_sell = my_coin.coin_name
    status_list = []
    for coin_name in ticker_sell:
        avg_price = float(df.avg_buy_price)
        current_price = pyupbit.get_current_price(coin_name)
        ratio = (current_price - avg_price) / avg_price
        hoga = get_hoga_price(coin_name)
        balance = float(df.balance[0])
        expiration = False
        if not coin_name in ticker_predict: # predict sell ...
            #coin_name = my_coin.coin_name[1]
            df = my_coin[my_coin.coin_name == coin_name]
            df.reset_index(drop=True, inplace=True)
            time.sleep(0.1)
            result = []
        else:
            df_sell = sell_df[sell_df.coin_name == coin_name]
            currency_time = datetime.now()
            start_date_ask = np.min(df_sell.start_date_ask)
            end_date_ask = np.max(df_sell.end_date_ask)
            max_price = np.mean(df_sell.sell_price_min) - 1.96 * np.sqrt(
                np.std(df_sell.sell_price_min) / len(df_sell.sell_price_min))
            updown_levels = np.mean(df_sell.influence_m)
            volume_levels = np.mean(df_sell.influence_v)
            if currency_time > end_date_ask:
                 expiration = True
        try:
            if bool(ratio < (-cutoff)) | bool(expiration):  # 손절 / period end
                print("코인 손절 요청: " + coin_name)
                price = current_price
            elif ratio > 0.001:
                print("코인 익절 요청: " + coin_name)
                if ratio > benefit:
                    min_price = current_price
                else:
                    min_price = max(avg_price * (1 + benefit), current_price)
                ho = list(hoga.get('ask_price')[hoga.get('ask_price') >= min_price])
                if len(ho) > 0:
                    price = ho[0]
            else:
                min_price = min(avg_price * (1 + benefit), current_price)
                ho = list(hoga.get('ask_price')[hoga.get('ask_price') >= min_price])
                if len(ho) > 0:
                    price = ho[0]
            res = upbit.sell_limit_order(coin_name, price, balance)
            time.sleep(0.1)
            if len(res) > 2:
                print(coin_name + "을 " + str(ho) + "에" + " 매도요청합니다.")
                save_uuid(res)
                status = 'res_bid'
                status_list.append(status)
        except:
            print('매도 error: ' + coin_name)
    sell_df.reset_index(drop=True, inplace=True)
    status_df = pd.DataFrame(status_list, columns=['status'], index=[0])
    result = pd.concat([status_df, sell_df], axis=1)
    return result

    # 판매 가격 결정하기...
def execute_buy_schedule(upbit, mdf, investment):
    money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
    if money < 5000:
        print("주문금액 부족")
        result = mdf
    else:
        # mdf = buy_df
        mdf.reset_index(drop=True, inplace=True)
        status_list = []
        tickers = list(set(mdf.coin_name)) # 구매신청
        for coin_name in tickers:
            #coin_name ='KRW-XRP'
            df_buy = mdf[mdf.coin_name == coin_name]
            status = 'wait'
            currency_time = datetime.now()
            start_date_ask = np.min(df_buy.start_date_ask)
            end_date_ask = np.max(df_buy.end_date_ask)
            min_price = np.mean(df_buy.buy_price_max) - 1.96 * np.sqrt(np.std(df_buy.buy_price_max)/len(df_buy.buy_price_max))
            max_price = np.mean(df_buy.sell_price_min) - 1.96 * np.sqrt(np.std(df_buy.sell_price_min)/len(df_buy.sell_price_min))
            updown_levels = np.mean(df_buy.influence_m)
            volume_levels = np.mean(df_buy.influence_v)
            levels = updown_levels + volume_levels
            # 구매 가격 결정하기
            try:
                money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
                time.sleep(0.1)
            except:
                money = investment
            if money < 5000:
                pass
            else:
                last_investment = min(investment, money)
                hoga = get_hoga_price(coin_name)
                default = min_price
                add = 1+levels/10
                if bool(min(hoga.get('bid_price')) < min_price * add):
                    print(coin_name+"은 구매 계획에 도달하지 않았다.")
                else:
                    print(coin_name+ "은 구매 계획에 도달했다.")
                    ho = list(hoga.get('bid_price')[hoga.get('bid_price') >= min_price])
                    if len(ho)>=3:
                        price_set = ho[-3:]
                    else:
                        price_set = ho

                    for price in price_set:
                        count = investment / price
                        try:
                            res = upbit.buy_limit_order(coin_name, price, count)
                            time.sleep(0.1)
                            if len(res) > 2:
                                print(coin_name +"을 "+str(price)+"에" + " 구매요청합니다.")
                                save_uuid(res)
                                status = 'res_ask'
                                status_list.append(status)
                        except:
                            print('매수 error: ' + coin_name)
        df_buy.reset_index(drop=True, inplace=True)
        status_df = pd.DataFrame(status_list, columns=['status'], index=[0])
        result = pd.concat([status_df, df_buy], axis=1)
    return result
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
def coin_predict(default_df):
    df = default_df[-5:]
    df.reset_index(inplace=True)
    buy_price = min(df.low)
    buy_price_variance = np.mean(df.length_low)
    sell_price = max(df.high)
    sell_price_variance = np.mean(df.length_high)

    buy_price_max = buy_price + buy_price_variance
    sell_price_min = sell_price - sell_price_variance
    #weight = df.index[np.mean(df.length_mid)*2 < df.length_mid].tolist()
    influence_m = np.sum(np.mean(df.length_mid)*2 < df.length_mid)
    influence_v = np.sum(np.mean(df.volume)*1.5 < df.volume)

    check_buy = check_buy_case(df)
    check_sell = check_sell_case(df)
    # 상태
    status = 'normal'
    result = {'check_buy':check_buy,'check_sell':check_sell,'buy_price_max':buy_price_max, 'sell_price_min':sell_price_min, 'status':status, 'influence_m': influence_m,
              'influence_v':influence_v}
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
    default_df = []
    for i in range(len(default_res)-1):
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
    result = box_vervus(default_res)
    return result
def changed_ratio(df, price_currency):
    df_1 = df[-1:]
    changed_ratio = (df_1.close[0] - df_1.open[0]) / df_1.open[0]
    result = changed_ratio
    return result

def coin_information(coin_name, interval):
    price_currency = pyupbit.get_current_price(coin_name)
    df = pyupbit.get_ohlcv(coin_name, interval=interval, count=10)
    ratio = changed_ratio(df, price_currency)
    time.sleep(0.1)
    box = box_information(df) ### criteria_boxplot
    start_date_ask = datetime.now()
    end_date_ask = period_end_date(interval)
    start_date_bid = end_date_ask + timedelta(seconds=1)
    end_date_bid = start_date_bid + (end_date_ask - start_date_ask)
    result = box  ### coin_check
    result['start_date_ask'] = start_date_ask
    result['end_date_ask'] = end_date_ask
    result['start_date_bid'] = start_date_bid
    result['end_date_bid'] = end_date_bid
    result['price_changed_ratio'] = ratio
    result['interval'] = interval
    result['coin_name'] = coin_name
    return result
def coin_validation(upbit, tickers, interval):
    li = []
    for coin_name in tickers:
        #coin_name = tickers[1]
        res = coin_information(coin_name, interval)
        li.append(res)
    df = pd.DataFrame(li)
    result = df
    merge_df(result)
    return result
def save_uuid(dict):
    # uuid -> dataframe, 기존의 uuid를 불러온다. uuid를 저장한다
    df = pd.DataFrame(dict, index=[0])
    df = df.dropna(axis=0)

    dt_now = datetime.now()
    now = dt_now.strftime('%Y%m%d%H%M%S')
    name = 'reservation_'+now+'.json'
    directory = 'reservation'
    json_file = directory+'/'+name
    # 폴더생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        # 데이터 읽고 머지
    if os.path.isfile(directory):
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')
        ori_df = pd.DataFrame(ori_df)
        df = pd.concat([ori_df, df], axis=0)

    # 저장
    df.to_json(json_file, orient='table')
def load_uuid():
    # uuid -> dataframe, 기존의 uuid를 불러온다. uuid를 저장한다
    directory = 'reservation'
    ori_df= []
    if os.path.exists(directory):
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')
    else:
        print("예약 내역이 없습니다.")
    return ori_df
def reservation_cancel(upbit):
    df = load_uuid()
    if len(df) > 0:
        uuids = list(set(df.uuid))
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
    dt_now = datetime.now()
    now = dt_now.strftime('%Y%m%d')
    name = 'schedule_' + now + '.json'
    directory = 'schedule'
    json_file = directory + '/' + name
    # 폴더생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = df.dropna(axis=0)
    df = pd.DataFrame(df)
    if os.path.isfile(directory):
        name = os.listdir(directory)[-1]
        json_file = directory + '/' + name
        ori_df = pd.read_json(json_file, orient='table')
        ori_df = pd.DataFrame(ori_df, axis=0)
        df = pd.concat([ori_df, df], axis=0)
    else:
        df = df

    #데이터 읽고 머지
    df.to_json(json_file, orient='table')

    #저장
    result = df
    return result
#구매 평가 요소. 주 동향,

def generate_benefit_cutoff(new_df):
    count = np.sum(new_df.price_changed_ratio > 0)
    benefit = np.mean(new_df.price_changed_ratio>0)
    cutoff = np.mean(new_df.price_changed_ratio)
    result = {'up_count': count, 'benefit':benefit, 'cutoff':cutoff}
    return result

def coin_trade(upbit, interval, total_updown, investment):
    tickers = pyupbit.get_tickers(fiat="KRW")
    try:
        new_df = load_df()
    except:
        print("현황 분석 시작.")
        new_df = coin_validation(upbit, tickers, interval)
        # 저장된 기존 df 를 불러오고 현재 생성된 df를 합친다.
    print("현황 분석 완료.")

    a = generate_benefit_cutoff(new_df)
    benefit = a.get('benefit')
    cutoff = a.get('cutoff')
    result = a.get('up_count')
    print("총 코인 수: "+str(len(new_df))+"개, 현재 상승 코인수: "+str(result)+"개, 예상 수익률 : "+str(round(benefit,2)*100)+"%, 예상 손절 : "+str(round(cutoff,2)*100)+"%")
    buy_df = new_df[new_df.check_buy]
    buy_df.reset_index(drop=True, inplace=True)
    buy_df = execute_buy_schedule(upbit, buy_df, investment)
    sell_df = new_df
    sell_df = execute_sell_schedule(upbit, sell_df, cutoff, benefit)

    print("schedual generate 시작")
    st = time.time()
    new_df = coin_validation(upbit, interval)
    et = time.time()
    diff = et - st
    print("schedual generate 종료(" + interval + "): " + str(round(diff, 1)) + '초')

    reservation_cancel(upbit)
    coin_validation(upbit, tickers, interval)

    return result


# selection

# input 1번 불러오면 되는 것들
if __name__ == '__main__':
    intervals = ["month", "week", "day", "minute240", "minute60", "minute30", "minute10", 'minute5']
    #access_key = '13OWmDwccuUleOzGq5Axg3PmfW1KoFO5igDyuSYM'
    access_key = '8o1RiU3sdJDga1jPx34ovI2f5agvPwIw9LAQzNgK'
    #secret_key = 'DN2uMoPwDGF7sa3lbaR4OYGAFg9UNom8erlofCox'
    secret_key = 'JUMqnCfnmWxjAqHC04cvqf4bs6JuwbBOHJv58I1y'

    upbit = pyupbit.Upbit(access_key, secret_key)
    interval = intervals[7]
    investment = 10000
    cutoff = 0.015
    total_updown = []
    while True:
        try:
            res = coin_trade(upbit, interval, total_updown, investment)
            total_updown.append(res)
        except:
            pass
