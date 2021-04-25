import numpy as np
import pandas as pd
import pyupbit
import time
###########
# input
############
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

def criteria_updown(df):
    avg_open = np.mean(df.open[0:len(df)-1])
    avg_high = np.mean(df.high[0:len(df)-1])
    avg_low = np.mean(df.low[0:len(df)-1])
    avg_close = np.mean(df.close[0:len(df)-1])
    avg_volume = np.mean(df.volume[0:len(df)-1])
    open_rate = df.open[len(df)-1]/df.open[len(df)-2]
    if open_rate < 1:
        open_status = 'down'
    elif open_rate > 1:
        open_status = 'up'
    else:
        open_status = 'normal'
    volume_rate = df.volume[len(df)-1]/df.volume[len(df)-2]
    if volume_rate < 1 :
        volume_status = 'down'
    elif volume_rate > 1:
        volume_status = 'up'
    else:
        volume_status = 'normal'
    result = {'open_status': open_status, 'open_rate': open_rate, 'volume_status': volume_status, 'volume_rate': volume_rate, 'avg_open':avg_open, 'avg_high':avg_high, 'avg_low':avg_low, 'avg_close':avg_close, 'avg_volume':avg_volume}
    return result

def criteria_boxplot(df):
    result = []
    avg_result =[]
    for i in range(len(df)):
        avg = criteria_updown(df[-i-1:])
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
        res = {'chart_name': chart_name, 'volume': df['volume'][i],'type':type, 'length': diff_length, 'high_length': diff_high, 'low_length': diff_low, 'middle_length': diff_middle}
        result.append(res)
    new_df = pd.DataFrame(result)
    new_df2 = pd.DataFrame(avg_result)
    result_df = pd.concat([new_df, new_df2], axis=1)
    result_df.set_index(df.index, inplace=True)
    return result_df

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
    elif bool(np.sum(box.type == 'blue')>=4) & bool(box.chart_name[idx-1] in ['doji_cross','spinning_tops']):
        result = True
    return result

def check_sell_case(box):
    idx = len(box)
    result = False
    if bool(np.sum(box.type == 'red')>=3) & bool(box.chart_name[idx-1] in ['stone_cross','lower_tail_pole_umbong','upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(box.type == 'red')>=4) & bool(box.chart_name[idx-2] in ['longbody_yangbong','shortbody_yangbong']):
        result = True
    elif bool(np.sum(box.type == 'red')>=4) & bool(box.chart_name[idx-1] in ['spinning_tops','doji_cross']):
        result = True
    elif bool(box.volume_status[idx - 2] == 'up') & bool(box.open_status[idx - 2] == 'up') & bool(box.volume_status[idx - 1] == 'down') & bool(box.open_status[idx - 2] == 'down'):
        result = True
    return result

def coin_check(box):
    idx = len(box)
    volume_levels = np.sum(box.volume_rate > 1) # 관심도가 증가 구매/판매 우선순위
    updown_levels = np.sum(box.open_rate > 1)   # 상승 강도 0~5 price 조정
    avg_volume_rate = np.mean(box.volume_rate)
    avg_open_rate = np.mean(box.open_rate)
    check_buy = check_buy_case(box)
    check_sell = check_sell_case(box)
    result = {'volume_levels': volume_levels,'avg_volume_rate':avg_volume_rate,'avg_open_rate': avg_open_rate, 'updown_levels': updown_levels, 'check_buy': check_buy, 'check_sell': check_sell}
    return result

def coin_info(coin_name, interval):
    df = pyupbit.get_ohlcv(coin_name, interval=interval, count=5)
    time.sleep(0.1)
    box = criteria_boxplot(df)
    result = coin_check(box)
    result['coin_name'] = coin_name
    return result

#호가를 이용
def get_hoga_price(coin_name, no):
    # input : coint_name
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    ask_price = df_orderbook.ask_price[no]
    bid_price = df_orderbook.bid_price[no]
    result = {'ask_price': ask_price, 'bid_price': bid_price}
    return result

# 현재 내 코인 정보
def my_coin(upbit, remove_tickers):
    my_balances = pd.DataFrame(upbit.get_balances())
    if len(my_balances)<2:
        print('판매 가능한 코인이 없습니다.')
    else:
        my_balances['coin_name'] = my_balances.unit_currency + '-' + my_balances.currency
        for remove_coine in remove_tickers:
            my_balances = my_balances[my_balances.currency != remove_coine]
        my_balances.reset_index(drop=True, inplace=True)
        my_balances = my_balances[pd.to_numeric(my_balances.balance) > 0]
        my_balances = my_balances[pd.to_numeric(my_balances.avg_buy_price) > 0]
    return my_balances

def ask_coin(upbit, coin_name, investment, no):
    money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
    hoga_price = get_hoga_price(coin_name, no)
    last_investment = min(investment, money)
    price = hoga_price.get('ask_price')
    count = last_investment / price
    try:
        result = upbit.buy_limit_order(coin_name, price, count)
        print('상승 예측 / 매수 요청: ' + coin_name)
    except:
        print('매수 error: ' + coin_name)
    return result

def bid_coin(upbit, coin_name, my_balances, cutoff, benefit, no):
    df = my_balances[my_balances.coin_name == coin_name]
    df.reset_index(drop=True, inplace=True)
    avg_price = pd.to_numeric(df.avg_buy_price)[0]
    hoga_price = get_hoga_price(coin_name, no)
    price = hoga_price.get('bid_price')
    current_price = pyupbit.get_current_price(coin_name)
    ratio = (current_price - avg_price) / avg_price
    balance = float(df.balance[0])
    result = []
    try:
        if ratio < (-cutoff): # 손절
            print("코인 손절 요청: "+coin_name)
            result = upbit.sell_limit_order(coin_name, min(price, current_price), balance)
        elif ratio > benefit:
            print("코인 익절 요청: "+coin_name)
            result = upbit.sell_limit_order(coin_name, max(price, current_price), balance)
        else:
            result = {'etc': 'wait coin'}
    except:
        print('매도 error: ' + coin_name)
    return result

def bid_coin_set(upbit, coin_name, remove_tickers, cutoff, benefit, no):
    my_balances = my_coin(upbit, remove_tickers)
    a = np.sum(my_balances.coin_name == coin_name)
    money = pd.to_numeric(upbit.get_balances()[0].get('balance'))
    result =[]
    if bool(a > 1): # 탐색중 판매 신호가 발생
        print("하락 예측 / 매도 요청: "+ coin_name)
        result = bid_coin(upbit, coin_name, my_balances, cutoff, benefit)
    return result

def check_my_price(upbit, remove_tickers):
    my_balances = my_coin(upbit, remove_tickers)
    currency_price = list()
    for coin_name in my_balances.coin_name:
        try:
            res = pyupbit.get_current_price(coin_name)
            time.sleep(0.1)
            currency_price.append(res)
        except:
            print('현재가 조회 실패: '+coin_name)
    KRW = pd.to_numeric(upbit.get_balances()[0].get('balance'))
    my_balances['tot_price'] = pd.to_numeric(my_balances['balance']) * pd.to_numeric(my_balances['avg_buy_price'])
    my_balances['currecy_price'] = pd.to_numeric(currency_price)
    my_balances['tot_currecy_price'] = pd.to_numeric(my_balances['balance']) * pd.to_numeric(currency_price)

    my_balances['pnl'] = (pd.to_numeric(my_balances['currecy_price']) - pd.to_numeric(my_balances['avg_buy_price'])) * pd.to_numeric(my_balances['balance'])
    # coin_validation = my_balances.sort_values('pnl', ascending= False)
    a = np.sum(my_balances['tot_price'])
    b = np.sum(my_balances['tot_currecy_price'])
    my_money_value = KRW + b
    revenue = (b - a)
    revenue_ratio = revenue/a
    result = {'my_money_value': round(my_money_value,2),'KRW': round(KRW,0), 'my_revenue': round(revenue,0), 'my_revenue_ratio': round(revenue_ratio,2), 'tot_price': round(a,0), 'tot_currency_price': round(b,0)}
    return result

def main(upbit, interval, remove_tickers, investment, cutoff, benefit, ask_no,bid_no):
    print("시작")
    validation = list()
    start_my_price = check_my_price(upbit, remove_tickers)
    start_time = time.time()
    start_my_price['time'] = 0
    validation.append(start_my_price)

    while True:
        tickers = pyupbit.get_tickers(fiat="KRW")
        tickers = [w for w in tickers if np.sum([str('KRW-'+c) in w for c in remove_tickers]) == 0]
        result = []
        buy_list = []
        sell_list = []
        # 구매/ 판매 시도
        for coin_name in tickers:
            coin_name = tickers[0]
            try:
                money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
                my_balances = my_coin(upbit, remove_tickers)
                sell_tickers = list(my_balances.coin_name)
                res = coin_info(coin_name, interval)
                result.append(res)
            except:
                print("coin_info")

            a = bool(res.get('avg_volume_rate')>1) & bool(res.get('avg_open_rate') > 1)
            b = bool(coin_name in sell_tickers)
            if bool(a | res.get('check_buy')) & bool(money > 5000):
                try:
                    ask_result = ask_coin(upbit, coin_name, investment, ask_no)
                    if len(list(ask_result)) > 2:
                        print("매수 요청: " + coin_name)
                        buy_list.append(ask_result)
                        time.sleep(0.1)
                    else:
                        print(ask_result.get('error'))
                except:
                    print("매수 error :" + coin_name)
            elif a | res.get('check_sell'):
                try:
                    bid_result = bid_coin_set(upbit, coin_name, remove_tickers, cutoff, benefit, bid_no)
                    if len(list(bid_result)) > 2:
                        sell_list.append(bid_result)
                        time.sleep(0.1)
                except:
                    print("매도(하락) error :"+coin_name)
            elif b:
                try:
                    bid_result = bid_coin(upbit, coin_name, my_balances, cutoff, benefit, bid_no)
                    if len(list(bid_result)) > 2:
                        sell_list.append(bid_result)
                        time.sleep(0.1)
                except:
                    print('매도 error: ' + coin_name)

        # 예약 취소
        buying_reservation_coin_cancel(upbit, sell_list)
        buying_reservation_coin_cancel(upbit, buy_list)
        end_my_price = check_my_price(upbit, remove_tickers)

        end_time = time.time()
        diff_time = round((end_time - start_time),1)
        end_my_price['run_time'] = diff_time
        print('수익 :'+str(end_my_price.get('KRW')+end_my_price.get('my_revenue'))+"("+str(end_my_price.get('my_revenue_ratio')))
        validation.append(end_my_price)
    return result

#접속
def connection_upbit(case):
    #case = 'work'
    if case == 'home':
        access_key = 'DXqzxyCQkqL9DHzQEyt8pK5izZXU03Dy2QX2jAhV'
        secret_key = 'x6ubxLyUVw03W3Lx5bdvAxBGWI7MOMJjblYyjFNo'
    elif case == 'work':
        access_key = 'Ae8p07023M16i2b0ONBRs0YVf2yBUcvvBpKjZrT5'
        secret_key = 'U1Fo9YcEzq9SR6T31JbXsvyOWREy4kBYs5PGx2Hs'
    upbit = pyupbit.Upbit(access_key, secret_key)
    return upbit

#매수 전체 취소 프로세스
def buying_reservation_coin_cancel(upbit, variable_list):
    if len(variable_list)==1:
        uuids = variable_list[0].get('uuid')
        try:
            for uuid in uuids:
                upbit.cancel_order(uuid)
                time.sleep(0.1)
        except:
            print("예약된 구매 리스트가 없습니다.")
    else:
        try:
            uuids = list(pd.DataFrame(variable_list)['uuid'].dropna())
            for uuid in uuids:
                upbit.cancel_order(uuid)
                time.sleep(0.1)
        except:
            print( "예약된 구매 리스트가 없습니다.")
