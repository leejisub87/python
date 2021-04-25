import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#호가를 이용
def get_hoga_price(coin_name):
    # input : coint_name
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    ask_price = df_orderbook.ask_price
    bid_price = df_orderbook.bid_price
    result = {'ask_price': ask_price, 'bid_price': bid_price}
    return result

def execute_sell_schedule(upbit, coin_name, df_sell, investment):
    currency_time = datetime.now()
    df = df_sell[currency_time <= df_sell.end_date_bid]
    df = df_sell[currency_time > df_sell.start_date_bid]
    min_price = df.range_sell_min
    max_price = df.range_sell_max
    updown_levels = df.updown_levels
    volume_levels = df.volume_levels

    # 판매 가격 결정하기...



def execute_buy_schedule(upbit, coin_name, df_buy, investment):
    currency_time = datetime.now()
    df = df_buy[currency_time <= df_buy.end_date_ask]
    df = df_buy[currency_time > df_buy.start_date_ask]
    min_price = df.range_buy_min
    max_price = df.range_buy_max
    updown_levels = df.updown_levels
    volume_levels = df.volume_levels

    # 구매 가격 결정하기
    money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
    last_investment = min(investment, money)
    price_set = np.random.uniform(low=min_price, high=max_price, size=(3,)).astype(int)
    price_set.sort(1)
    price_weight = [0.2, 0.3, 0.5]
    result = []
    for price in price_set:
        try:
            count = investment / price / len(price_set)
            res = upbit.buy_limit_order(coin_name, price, count)
            time.sleep(0.1)
            if len(res)>2:
                result = res.get('uuid')
                return result
        except:
            print('매수 error: ' + coin_name)

# start - coin_info
##coin check
def check_buy_case(box):
    idx = len(box)
    result = False
    if bool(np.sum(box.type == 'blue')>=3) & bool(box.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong']):
        result = True
    elif bool(np.sum(box.type == 'blue')>=3) & bool(box.chart_name[idx-1] in ['dragonfly_cross', 'upper_tail_pole_yangong']):
        result = True
    elif bool(box.chart_name[idx-1] in ['pole_yangbong','longbody_yangbong']):
        result = True
    elif bool(np.sum(box.type == 'blue')>=3) & bool(box.chart_name[idx-1] == 'pole_umbong'):
        result = True
    elif bool(np.sum(box.type == 'blue')>=3) & bool(box.chart_name[idx-1] in ['doji_cross','spinning_tops']):
        result = True
    return result
def check_sell_case(box):
    idx = len(box)
    result = False
    if bool(np.sum(box.type == 'red')>=3) & bool(box.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong','upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(box.type == 'red')>=3) & bool(box.chart_name[idx-2] in ['longbody_yangbong','shortbody_yangbong']):
        result = True
    elif bool(np.sum(box.type == 'red')>=3) & bool(box.chart_name[idx-1] in ['spinning_tops','doji_cross']):
        result = True
    elif bool(box.volume_status[idx - 2] == 'up') & bool(box.open_status[idx - 2] == 'up') & bool(box.volume_status[idx - 1] == 'down') & bool(box.open_status[idx - 2] == 'down'):
        result = True
    return result
def coin_check(box):
    idx = len(box)
    volume_levels = np.sum(box.volume_rate > 1) # 관심도가 증가 구매/판매 우선순위
    updown_levels = np.sum(box.open_rate > 1)   # 상승 강도 0~4 price 조정
    avg_volume_rate = np.mean(box.volume_rate)

    n = len(box.avg_open)
    avg_open = np.mean(box.avg_open)
    avg_low = np.mean(box.avg_low)
    avg_close = np.mean(box.avg_close)
    avg_high = np.mean(box.avg_high)
    std_open = np.std(box.std_open)
    std_low = np.std(box.std_low)
    std_close = np.std(box.std_close)
    std_high = np.std(box.std_high)

    avg_open_rate = np.mean(box.open_rate)
    avg_close_rate = np.mean(box.close_rate)

    range_buy_min = avg_low - 1.96 * std_low / n
    range_buy_max = min(avg_open, avg_close) + 1.96 * min(std_open, std_close) / n
    range_sell_min = max(avg_open, avg_close) - 1.96 * min(std_open, std_close) / n
    range_sell_max = avg_high +1.96 * std_high / n

    check_buy = check_buy_case(box)
    check_sell = check_sell_case(box)
    result = {
        'range_buy_min':range_buy_min,
        'range_buy_max':range_buy_max,
        'range_sell_min':range_sell_min,
        'range_sell_max':range_sell_max,
        'check_buy': check_buy,
        'check_sell': check_sell,
        'open_price':box.open_price[-1],
        'close_price':box.close_price[-1],
        'volume_levels': volume_levels,
        'avg_volume_rate':avg_volume_rate,
        'avg_open_rate': avg_open_rate,
        'avg_close_rate': avg_close_rate,
        'updown_levels': updown_levels
    }
    return result

##criteria_boxplot
def criteria_updown(df):
    avg_open = np.mean(df.open[0:len(df) - 1])
    avg_close = np.mean(df.close[0:len(df) - 1])
    avg_volume = np.mean(df.volume[0:len(df) - 1])
    avg_high = np.mean(df.high[0:len(df) - 1])
    avg_low = np.mean(df.low[0:len(df) - 1])
    std_open = np.std(df.open[0:len(df) - 1])
    std_close = np.std(df.close[0:len(df) - 1])
    std_volume = np.std(df.volume[0:len(df) - 1])
    std_high = np.std(df.high[0:len(df) - 1])
    std_low = np.std(df.low[0:len(df) - 1])
    a = df[0:2]
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
    result = {
        'volume_rate' : volume_rate,
        'open_rate' : open_rate,
        'close_rate' :close_rate,
        'volume_status': volume_status,
        'open_status': open_status,
        'close_status': close_status,
        'avg_open' : avg_open,
                'avg_close' : avg_close,
                'avg_volume' : avg_volume,
                'avg_high' : avg_high,
                'avg_low' : avg_low,
                'std_open' : std_open,
                'std_close' : std_close,
                'std_volume' : std_volume,
                'std_high' : std_high,
                'std_low' : std_low,
              'open_price' : df.open[-1],
              'close_price' : df.close[-1]}
    return result
def criteria_boxplot(df):
    result = []
    avg_result = []
    for i in range(len(df)):
        avg = criteria_updown(df[-i - 1:])
        avg_result.append(avg)
        value = df['close'][i] - df['open'][i]
        if value >= 0:
            type = "red"
            diff_high = df['high'][i] - df['close'][i]
            diff_middle = value
            diff_low = df['open'][i] - df['low'][i]
            diff_length = df['high'][i] - df['low'][i]
        else:
            type = "blue"
            diff_high = df['high'][i] - df['open'][i]
            diff_middle = - value
            diff_low = df['close'][i] - df['low'][i]
            diff_length = df['high'][i] - df['low'][i]
        chart_name = criteria_chart_name(type, diff_high, diff_middle, diff_low)
        res = {'chart_name': chart_name, 'volume': df['volume'][i], 'type': type, 'length': diff_length,
               'high_length': diff_high, 'low_length': diff_low, 'middle_length': diff_middle}
        result.append(res)
    new_df = pd.DataFrame(result)
    new_df2 = pd.DataFrame(avg_result)
    result_df = pd.concat([new_df, new_df2], axis=1)
    result_df.set_index(df.index, inplace=True)
    return result_df
def criteria_chart_name(type, high, middle, low):
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
# end - coin_info

def coin_info(upbit, coin_name, interval):
    res = pyupbit.get_current_price(coin_name)
    df = pyupbit.get_ohlcv(coin_name, interval='month', count=5)
    time.sleep(0.1)
    box = criteria_boxplot(df) ### criteria_boxplot
    result = coin_check(box)   ### coin_check
    #intervals = ["month", "days", "minute240", "minute60", "minute30", "minute10", "minute5", "minute3", "minute1"]
    start_date_ask = datetime.now()
    end_date_ask = period_end_date("month")
    start_date_bid = end_date_ask + timedelta(seconds = 1)
    end_date_bid = start_date_bid + (end_date_ask - start_date_ask)
    result['start_date_ask'] = start_date_ask
    result['end_date_ask'] = end_date_ask
    result['start_date_bid'] = start_date_bid
    result['end_date_bid'] = end_date_bid
    result['interval'] = interval
    result['coin_name'] = coin_name
    return result
def df_schedule(upbit, tickers, interval):
    schedule_list = []
    for coin_name in tickers:
        try:
            res = coin_info(upbit, coin_name, interval)
            schedule_list.append(res)
        except:
            pass
    result = pd.DataFrame(schedule_list)
    return result
def generate_schedule(access_key, secret_key, investment):
    intervals = ["month", "week", "day", "minute240", "minute60", "minute30","minute10"]
    upbit = pyupbit.Upbit(access_key, secret_key)
    tickers = pyupbit.get_tickers(fiat="KRW")
    my_coin = pd.DataFrame(upbit.get_balances())

    interval = intervals[0]
    coin_name = tickers[0]
    df = df_schedule(upbit, tickers, interval)
    # 저장된 기존 df 를 불러오고 현재 생성된 df를 합친다.
    # 합치고
    df_buy = df[df['check_buy']]
    df_sell = df[df['check_sell']]
    buy_uuids = []
    if len(df_buy) > 0:
        uuid = execute_buy_schedule(upbit, coin_name, df_buy, investment)
        buy_uuids.append(uuid)
    else:
        pass
    sell_uuisd = []
    if len(df_sell) > 0:
        uuid = execute_sell_schedule(upbit, coin_name, df_sell, investment)
        buy_uuids.append(uuid)
    else:
        pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    access_key = 'DXqzxyCQkqL9DHzQEyt8pK5izZXU03Dy2QX2jAhV'
    secret_key = 'x6ubxLyUVw03W3Lx5bdvAxBGWI7MOMJjblYyjFNo'
    investment = 10000
    generate_schedule(access_key, secret_key, investment)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
