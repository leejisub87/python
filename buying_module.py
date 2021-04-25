import numpy as np
import pandas as pd
import pyupbit
import time
pd.set_option('display.max_columns', 20)
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
    if len(df) > 1:
        open_rate = df.open[1]/df.open[0]
        volume_rate = df.volume[1] / df.volume[0]
    else:
        open_rate = 0
        volume_rate = 0
    if open_rate < 1:
        open_status = 'down'
    elif open_rate > 1:
        open_status = 'up'
    else:
        open_status = 'normal'

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
    if bool(np.sum(box.type == 'blue')>=4) & bool(box.chart_name[idx-1] in ['stone_cross', 'lower_tail_pole_umbong', 'dragonfly_cross', 'upper_tail_pole_yangong']):
        result = True
    elif bool(box.chart_name[idx-1] in ['pole_yangbong','longbody_yangbong']):
        result = True
    elif bool(np.sum(box.type == 'blue')>=4) & bool(box.chart_name[idx-1] in ['pole_umbong', 'doji_cross','spinning_tops']):
        result = True
    return result

def check_sell_case(box):
    idx = len(box)
    result = False
    if bool(np.sum(box.type == 'red')>=4) & bool(box.chart_name[idx-1] in ['stone_cross', 'lower_tail_pole_umbong', 'upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(box.type == 'red')>=4) & bool(box.chart_name[idx-1] in ['longbody_yangbong', 'shortbody_yangbong']):
        result = True
    elif bool(np.sum(box.type == 'red')>=4) & bool(box.chart_name[idx-1] in ['spinning_tops', 'doji_cross']):
        result = True
    elif bool(box.volume_status[idx - 2] == 'up') & bool(box.open_status[idx - 2] == 'up') & bool(box.volume_status[idx - 1] == 'down') & bool(box.open_status[idx - 2] == 'down'):
        result = True
    return result

def coin_check(box):
    idx = len(box)
    volume_levels = np.sum(box.volume_rate > 1) # 관심도가 증가 구매/판매 우선순위
    updown_levels = np.sum(box.open_rate > 1)   # 상승 강도 0~4 price 조정
    avg_volume_rate = np.mean(box.volume_rate)
    avg_open_rate = np.mean(box.open_rate)
    check_buy = check_buy_case(box)
    check_sell = check_sell_case(box)
    buy_price = np.mean(box.avg_low) + 1.95 * np.std(box.avg_low)/5
    sell_price = np.mean(box.avg_high) - 1.95 * np.std(box.avg_high)/5
    result = {'diff_middle': buy_price, 'sell_price': sell_price,'buy_price': buy_price, 'volume_levels': volume_levels, 'avg_volume_rate': avg_volume_rate, 'avg_open_rate': avg_open_rate, 'updown_levels': updown_levels, 'check_buy': check_buy, 'check_sell': check_sell}
    return result

def coin_info(coin_name, interval):
    df = pyupbit.get_ohlcv(coin_name, interval=interval, count=5)
    time.sleep(0.1)
    box = criteria_boxplot(df)
    result = coin_check(box)
    result['coin_name'] = coin_name
    return result

#호가를 이용
def get_hoga_price(coin_name, price, type):
    # input : coint_name
    orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    if type == 'ask':
        bid_price = df_orderbook.bid_price
        idx = np.where(bid_price < price)
        idx2 = list(idx[0])
        result = list(bid_price[idx2])
    if type == 'bid':
        ask_price = df_orderbook.ask_price
        idx = np.where(ask_price > price)
        idx2 = list(idx[0])
        result = list(ask_price[idx2])
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

def ask_coin(upbit, coin_name, ask_price, investment):
    money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
    hoga_price = get_hoga_price(coin_name, ask_price, "ask")
    current_price = pyupbit.get_current_price(coin_name)
    last_investment = min(investment, money)
    ask_list = []
    for price in hoga_price:

        count = last_investment / price
        try:
            result = upbit.buy_limit_order(coin_name, min(price, current_price), count)
            time.sleep(0.01)
            print('상승 예측 / 매수 요청: ' + coin_name)
            if len(result)>2:
                uuid = result.get('uuid')
                ask_list.append(uuid)
            else:
                print("service error")
        except:
            print('매수 error: ' + coin_name)
    return ask_list

def bid_coin(upbit, coin_name, bid_price, my_balances, cutoff, benefit):
    df = my_balances[my_balances.coin_name == coin_name]
    df.reset_index(drop=True, inplace=True)
    avg_price = pd.to_numeric(df.avg_buy_price)[0]
    hoga_price = get_hoga_price(coin_name, bid_price, "bid")
    if len(hoga_price)>=3:
        hoga_price = hoga_price[0:3]
    current_price = pyupbit.get_current_price(coin_name)
    ratio = (current_price - avg_price) / avg_price
    balance = float(df.balance[0])
    if not hoga_price:
        balance_n = balance
        hoga_price = avg_price
    else:
        balance_n = balance / len(hoga_price)
    bid_list = []
    if ratio < (-cutoff): # 손절
        try:
            result = upbit.sell_limit_order(coin_name, current_price, balance)
            print("코인 손절 요청: "+coin_name)
            if len(result) > 2:
                uuid = result.get('uuid')
                bid_list.append(uuid)
            else:
                print("service error")
        except:
            print("service error : 손절 시도")
    elif ratio > benefit:
        for price in hoga_price:
            try:
                result = upbit.sell_limit_order(coin_name, max(current_price, price), balance_n)
                print("코인 익절 요청: "+coin_name)
                if len(result) > 2:
                    uuid = result.get('uuid')
                    bid_list.append(uuid)
                else:
                    print("service error")
            except:
                print("service error : 익절 시도")
    else:
        cc = np.where(hoga_price > avg_price * 1.2)[0]
        if len(cc) > 0:
            no = max(cc)
            result = upbit.sell_limit_order(coin_name, hoga_price[no], balance)
            if len(result) > 2:
                uuid = result.get('uuid')
                bid_list.append(uuid)
            else:
                print("service error")
        else:
            print("대기")
    return bid_list

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

def get_coin_selection(upbit):
    start_time = time.time()
    tickers = pyupbit.get_tickers(fiat="KRW")
    my_balances = []
    for coin_name in tickers:
        res = coin_info(coin_name, interval="minute10")
        my_balances.append(res)
    df = pd.DataFrame(my_balances).sort_values('avg_volume_rate', ascending = False)
    df = df[df.avg_volume_rate > 1]
    df = df[df.updown_levels >= 1]
    df = df[df.volume_levels >= 1]
    result = df
    end_time = time.time()
    diff_time = end_time - start_time
    print("후보 산출시간: "+str(diff_time)+"초")
    return result

def main(access_key, secret_key, interval, remove_tickers, investment, cutoff, benefit):
    print("시작")
    upbit = pyupbit.Upbit(access_key, secret_key)
    validation = list()
    start_my_price = check_my_price(upbit, remove_tickers)
    start_time = time.time()
    start_my_price['time'] = 0
    validation.append(start_my_price)
    favorite_coin = []
    n = 0
    m = 0
    while True:
        end_time = time.time()
        diff_time = end_time - start_time
        k = diff_time // 60*5
        while k >= n:
            n = n + 1
            while len(favorite_coin) == 0:
                favorite_coin = get_coin_selection(upbit)
        tickers = list(favorite_coin.coin_name)
        tickers = [w for w in tickers if np.sum([str('KRW-'+c) in w for c in remove_tickers]) == 0]
        start_time = time.time()
        diff_time = 0
        result = []
        ask_list = []
        bid_list = []

        # 구매
        for coin_name in tickers:
            try:
                money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
                res = coin_info(coin_name, interval)
                result.append(res)
            except:
                print("coin_info")

            a = bool(res.get('avg_volume_rate') > 1) & bool(res.get('avg_open_rate') > 1)
            if bool(a | res.get('check_buy')) & bool(money > 5000):
                try:
                    ask_price = res.get('buy_price')
                    ask_result = ask_coin(upbit, coin_name, ask_price, investment)
                    if not ask_result:
                        print("null")
                    else:
                        ask_list.append(ask_result)
                except:
                    print("매수 error :" + coin_name)

        # 판매
        my_balances = my_coin(upbit, remove_tickers)
        sell_tickers = list(my_balances.coin_name)
        for coin_name in sell_tickers:
            try:
                res = coin_info(coin_name, interval)
                bid_price = res.get('sell_price')
                bid_result = bid_coin(upbit, coin_name, bid_price, my_balances, cutoff, benefit)
                if not bid_result:
                    print("null")
                else:
                    bid_list.append(bid_result)
            except:
                print('매도 error: ' + coin_name)

        end_time = time.time()
        diff_time = end_time - start_time

        time.sleep(5)
        diff_time = end_time - start_time
        k = diff_time // 60 * 5
        while k >= m:
            m = m + 1
        # 예약 취소
            buying_reservation_coin_cancel(upbit, ask_list)
            buying_reservation_coin_cancel(upbit, bid_list)
        end_my_price = check_my_price(upbit, remove_tickers)
        end_time = time.time()
        diff_time = round((end_time - start_time),1)
        end_my_price['run_time'] = diff_time
        print('수익 :'+str(end_my_price.get('KRW')+end_my_price.get('my_revenue'))+"("+str(end_my_price.get('my_revenue_ratio'))+")")
        validation.append(end_my_price)
    return result



#매수 전체 취소 프로세스
def buying_reservation_coin_cancel(upbit, v_list):
    for i in range(len(v_list)):
        for j in range(len(v_list[i])):
            upbit.cancel_order(v_list[i][j])
            time.sleep(0.1)
