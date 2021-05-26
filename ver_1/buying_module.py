import numpy as np
import pandas as pd
import pyupbit
import time
import os
from datetime import datetime, timedelta
from sklearn import linear_model

pd.set_option('display.max_columns', 20)


###########
# input
############
def criteria_chart_name(type, high, middle, low):
    if bool(type == 'red') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle):  # 롱바디 양봉
        result = 'longbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle):  # 롱바디 음봉
        result = 'longbody_umbong'
    elif bool(type == 'red') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):  # 숏바디 양봉
        result = 'shortbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):  # 숏바디 음봉
        result = 'shortbody_umbong'
    elif bool(type == 'blue') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):  # 도지 십자
        result = 'doji_cross'
    elif bool(type == 'red') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):  # 릭쇼멘 도지
        result = 'rickshawen_doji'
    elif bool(type == 'blue') & bool(0 < middle * 5 < high) & bool(low == 0):  # 비석형 십자
        result = 'stone_cross'
    elif bool(type == 'red') & bool(0 < middle * 5 < low) & bool(high == 0):  # 잠자리형 십자
        result = 'dragonfly_cross'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle == 0):  #:포 프라이스 도지
        result = 'four_price_doji'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle > 0):  # 장대양봉
        result = 'pole_yangbong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low == 0) & bool(middle > 0):  # 장대음봉
        result = 'pole_umbong'
    elif bool(type == 'red') & bool(high > 0) & bool(low == 0) & bool(middle > 0):  # 윗꼬리 장대양봉
        result = 'upper_tail_pole_yangong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low > 0) & bool(middle > 0):  # 아랫꼬리 장대음봉
        result = 'lower_tail_pole_umbong'
    elif bool(type == 'red') & bool(high == 0) & bool(low > 0) & bool(middle > 0):  # 아랫꼬리 장대양봉
        result = 'lower_tail_pole_yangbong'
    elif bool(type == 'blue') & bool(high > 0) & bool(low == 0) & bool(middle > 0):  # 윗꼬리 장대음봉
        result = 'upper_tail_pole_umbong'
    elif bool(type == 'blue') & bool(middle * 5 < high) & bool(middle * 5 < low) & bool(middle > 0):  # 스피닝 탑스
        result = 'spinning_tops'
    elif bool(type == 'blue') & bool(0 < high <= middle) & bool(0 < low <= middle):  # 별형 스타
        result = 'start'
    else:
        result = 'need to name'
    return result


def criteria_boxplot(df):
    result = []
    avg_result = []
    for i in range(len(df)):
        avg = criteria_updown(df[-i - 1:].reset_index(drop=True))[0]
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


def coin_predict(res, avg_volume, min_value, max_value):
    # df = res[-5:]
    df = res
    df.reset_index(inplace=True, drop=True)
    v = list(df.rate_volume)
    influence_v = 0
    if (df.volume[len(df) - 1] >= avg_volume * 2):
        influence_v = 1
    elif (v[-1] >= avg_volume * 2) | (v[-2] >= avg_volume * 2):
        influence_v = 1
    elif (v[-1] >= avg_volume * 3) | (v[-2] >= avg_volume * 3):
        influence_v = 2

    check_buy = check_buy_case(df)
    check_buy_2 = check_buy_case2(df, min_value, max_value)
    check_sell = check_sell_case(df)
    # 상태
    result = {'influence_v': influence_v, 'check_buy': check_buy, 'check_buy_2': check_buy_2, 'check_sell': check_sell}
    return result


def check_buy_case(df):
    df = df[-5:].reset_index(drop=True)
    idx = len(df)
    updown = criteria_updown(df)
    updown = pd.DataFrame(updown)
    if bool(np.sum(updown.volume_status == 'down') >= 3):
        status = 'down'
    elif bool(np.sum(updown.volume_status == 'up') >= 3):
        status = 'up'
    else:
        status = 'normal'

    result = False
    if bool(status == 'down') & bool(df.chart_name[idx - 1] == 'shooting_star'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 1] == 'upper_tail_pole_yangong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 3] == 'lower_tail_pole_yangong') & bool(
            df.chart_name[idx - 2] == 'upper_tail_pole_yangong') & bool(
        df.chart_name[idx - 1] == 'lower_tail_pole_yangong'):
        result = True
    elif bool(df.chart_name[idx - 2] == 'longbody_umbong') & bool(df.chart_name[idx - 1] == 'shortbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 2] == 'longbody_umbong') & bool(
            df.chart_name[idx - 1] in ['shortbody_yangbong', 'rickshawen_doji', 'longbody_yangbong']):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 2] == 'longbody_umbong') & bool(
            df.chart_name[idx - 1] in ['longbody_yangbong', 'rickshawen_doji']):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 3] == 'longbody_umbong') & bool(
            df.chart_name[idx - 2] == 'shortbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 3] == 'longbody_umbong') & bool(
            df.chart_name[idx - 2] == 'doji_cross') & bool(
            df.chart_name[idx - 1] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 3] == 'longbody_umbong') & bool(
            df.chart_name[idx - 2] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'shortbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 3] == 'longbody_umbong') & bool(
            df.chart_name[idx - 2] == 'longbody_yangbong') & bool(
        df.chart_name[idx - 1] == 'shortbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.color == 'blue') >= 4) & bool(
            df.chart_name[idx - 2] == 'shooting_star') & bool(df.chart_name[idx - 1] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 2] == 'shooting_star') & bool(
            df.chart_name[idx - 1] == 'pole_yangbong'):
        result = True
    elif bool(status == 'down') & bool(df.chart_name[idx - 2] == 'longbody_umbong') & bool(
            df.chart_name[idx - 1] in ['longbody_yangbong', 'shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(
            df.chart_name[idx - 1] in ['stone_cross', 'lower_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(
            df.chart_name[idx - 1] in ['dragonfly_cross', 'upper_tail_pole_yangong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(np.sum(df.chart_name == 'pole_umbong') >= 2) & bool(
            df.chart_name[idx - 1] in ['shooting_star', 'upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'blue') >= 4) & bool(df.chart_name[idx - 1] in ['doji_cross', 'spinning_tops']):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong') >= 2) & bool(
            np.sum(df.chart_name == 'longbody_yangbong') >= 1) & bool(df.chart_name[idx - 1] in ['longbody_umbong']):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong') >= 2) & bool(
            np.sum(df.chart_name == 'shortbody_umbong') >= 1):
        result = True
    elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong') >= 2) & bool(
            np.sum(df.chart_name[4] in ['shooting_star', 'lower_tail_pole_umbong']) >= 1):
        result = True
    return result


def check_sell_case(df):
    df = df[-5:].reset_index(drop=True)
    idx = len(df)
    updown = criteria_updown(df)
    updown = pd.DataFrame(updown)
    result = False
    if bool(np.sum(updown.volume_status == 'down') >= 3) & bool(np.sum(updown.volume_status == 'down') >= 3):
        status = 'down'
    elif bool(np.sum(updown.volume_status == 'up') >= 3) & bool(np.sum(updown.volume_status == 'up') >= 3):
        status = 'up'
    else:
        status = 'normal'
    ####################################################
    #### 하락 반전형 캔들
    ## 유성형
    if bool(status == 'up') & bool(df.chart_name[idx - 1] == 'upper_tail_pole_yangong') & bool(
            df.color[idx - 1] == 'blue'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 1] == 'lower_tail_pole_yangbong') & bool(
            df.color[idx - 1] == 'red'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 1] == 'lower_tail_pole_umbong') & bool(
            df.color[idx - 1] == 'red'):
        result = True
    elif bool(status == 'up') & (bool(df.chart_name[idx - 2] in ['doji_cross', 'rickshawen_doji']) | bool(
            df.chart_name[idx - 1] in ['doji_cross', 'rickshawen_doji'])):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 2] == 'shortbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'longbody_yangbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 3] == 'upper_tail_pole_umbong') & bool(
            df.chart_name[idx - 2] == 'longbody_umbong') & bool(df.chart_name[idx - 1] == 'lower_tail_pole_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 2] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'doji_corss'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 2] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'upper_tail_pole_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 3] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 2] == 'longbody_umbong') & bool(df.chart_name[idx - 1] == 'shortbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 3] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 2] == 'shortbody_umbong') & bool(df.chart_name[idx - 1] == 'longbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 2] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'longbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 2] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 1] == 'longbody_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 3] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 2] == 'shortbody_yangbong') & bool(df.chart_name[idx - 1] == 'longbody_umbong'):
        result = True
    #########################################
    elif bool(status == 'up') & bool(df.chart_name[idx - 3] == 'longbody_yangbong') & bool(
            df.chart_name[idx - 2] == 'doji_corss') & bool(df.chart_name[idx - 1] == 'longbody_pole_umbong'):
        result = True
    elif bool(status == 'up') & bool(df.chart_name[idx - 3] == 'shortbody_yangbong') & bool(
            df.chart_name[idx - 2] == 'doji_corss') & bool(df.chart_name[idx - 1] == 'shortbody_umbong'):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(
            df.chart_name[idx - 1] in ['stone_cross', 'lower_tail_pole_umbong', 'upper_tail_pole_umbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(
            df.chart_name[idx - 2] in ['longbody_yangbong', 'shortbody_yangbong']):
        result = True
    elif bool(np.sum(df.color == 'red') >= 3) & bool(df.chart_name[idx - 1] in ['spinning_tops', 'doji_cross']):
        result = True
    return result


def criteria_updown(df):
    res = []
    if len(df) == 1:
        volume_status = 'normal'
        close_rate = 0
        open_status = 'normal'
        volume_rate = 0
        result = {'volume_status': volume_status, 'volume_rate': volume_rate, 'close_rate': close_rate,
                  'open_status': open_status}
        res.append(result)
    else:
        for i in range(len(df) - 1):
            a = df[i:i + 2].reset_index(drop=True)
            a.reset_index(drop=True, inplace=True)
            close_rate = a.close[1] / a.close[0]
            open_rate = a.open[1] / a.open[0]
            volume_rate = a.volume[1] / a.volume[0]
            if open_rate < 1:
                open_status = 'down'
            elif open_rate > 1:
                open_status = 'up'
            else:
                open_status = 'normal'
            if volume_rate < 1:
                volume_status = 'down'
            elif volume_rate > 1:
                volume_status = 'up'
            else:
                volume_status = 'normal'
            result = {'volume_status': volume_status, 'volume_rate': volume_rate, 'close_rate': close_rate,
                      'open_status': open_status}
            res.append(result)
    # t = pd.DataFrame(res).reset_index(drop=True)
    return res


def box_vervus(default_res):
    # default_df = default_res[-6:]
    default_df = default_res
    avg_volume = np.mean(default_df.volume)
    avg_close = np.mean(default_df.close)
    std_close = np.std(default_df.close)
    # 구매 단가 - 박스권 내 구매 가격
    min_value = avg_close - 1.7 * std_close
    # 판매 단가 - 박스권 내 구매 가격
    max_value = avg_close + 1.1 * std_close
    # 저항선 돌파 = 상승장
    up_levels = np.sum(default_df.high > avg_close + 2 * std_close)
    # 지지선 파괴 = 하락장
    down_levels = np.sum(default_df.low < avg_close - 2 * std_close)
    over_volume = np.sum((default_df[-2:].volume >= np.mean(default_df.volume) * 3) & (default_df[-2:].color=='red'))
    v = [0]
    for i in range(len(default_df) - 1):
        ddf = default_df[i:i + 2].reset_index(drop=True)
        odds_volume = ddf.volume[1] / ddf.volume[0]
        v.append(odds_volume)
    odds_volume = pd.DataFrame(v, columns=['rate_volume'])
    res = pd.concat([default_df.reset_index(drop=True), odds_volume.reset_index(drop=True)], axis=1)
    result = coin_predict(res, avg_volume, min_value, max_value)
    result['min_value'] = min_value
    result['max_value'] = max_value
    result['up_levels'] = up_levels
    result['down_levels'] = down_levels
    result['over_volume'] = over_volume
    return result


def box_create(df_1):
    df_1 = df_1.reset_index(drop=True)
    if bool(df_1.open[0] <= df_1.close[0]):
        df_1['color'] = 'red'
    else:
        df_1['color'] = 'blue'
    df_1['length_high'] = df_1.high[0] - np.max([df_1.close[0], df_1.open[0]])
    df_1['length_low'] = np.min([df_1.close[0], df_1.open[0]]) - df_1.low[0]
    df_1['length_mid'] = np.max([df_1.close[0], df_1.open[0]]) - np.min([df_1.close[0], df_1.open[0]])
    df_1['rate_mid'] = np.abs(df_1.close[0] - df_1.open[0]) / (df_1.open[0] + df_1.close[0]) * 2
    df_1['rate_high'] = df_1['length_high'] / (np.max([df_1.close[0], df_1.open[0]]) + df_1.high[0]) * 2
    df_1['rate_low'] = df_1['length_low'] / (np.min([df_1.close[0], df_1.open[0]]) + df_1.low[0]) * 2
    name = chart_name(df_1)
    df_1['chart_name'] = name
    return df_1


def chart_name(df_1):
    type = df_1.color[0]
    high = df_1.length_high[0]
    middle = df_1.length_mid[0]
    low = df_1.length_low[0]
    if bool(type == 'red') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle):  # 롱바디 양봉
        result = 'longbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 3 < middle) & bool(0 < low * 3 < middle):  # 롱바디 음봉
        result = 'longbody_umbong'
    elif bool(type == 'red') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):  # 숏바디 양봉
        result = 'shortbody_yangbong'
    elif bool(type == 'blue') & bool(0 < high * 1.2 < middle) & bool(0 < low * 1.2 < middle):  # 숏바디 음봉
        result = 'shortbody_umbong'
    elif bool(type == 'blue') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):  # 도지 십자
        result = 'doji_cross'
    elif bool(type == 'red') & bool(0 <= middle * 5 < high) & bool(0 < middle * 1.2 < low):  # 릭쇼멘 도지
        result = 'rickshawen_doji'
    elif bool(type == 'blue') & bool(0 < middle * 5 < high) & bool(low == 0):  # 비석형 십자
        result = 'stone_cross'
    elif bool(type == 'red') & bool(0 < middle * 5 < low) & bool(high == 0):  # 잠자리형 십자
        result = 'dragonfly_cross'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle == 0):  #:포 프라이스 도지
        result = 'four_price_doji'
    elif bool(type == 'red') & bool(high == 0) & bool(low == 0) & bool(middle > 0):  # 장대양봉
        result = 'pole_yangbong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low == 0) & bool(middle > 0):  # 장대음봉
        result = 'pole_umbong'
    elif bool(type == 'red') & bool(high > 0) & bool(low == 0) & bool(middle > 0):  # 윗꼬리 장대양봉
        result = 'upper_tail_pole_yangong'
    elif bool(type == 'blue') & bool(high == 0) & bool(low > 0) & bool(middle > 0):  # 아랫꼬리 장대음봉
        result = 'lower_tail_pole_umbong'
    elif bool(type == 'red') & bool(high == 0) & bool(low > 0) & bool(middle > 0):  # 아랫꼬리 장대양봉
        result = 'lower_tail_pole_yangbong'
    elif bool(type == 'blue') & bool(high > 0) & bool(low == 0) & bool(middle > 0):  # 윗꼬리 장대음봉
        result = 'upper_tail_pole_umbong'
    elif bool(type == 'blue') & bool(middle * 5 < high) & bool(middle * 5 < low) & bool(middle > 0):  # 스피닝 탑스
        result = 'spinning_tops'
    elif bool(type == 'blue') & bool(0 < high <= middle) & bool(0 < low <= middle):  # 별형 스타
        result = 'start'
    elif bool(type == 'blue') & bool(high > middle) & bool(low == 0) & bool(low < middle):  # 역망치 스타
        result = 'shooting_star'
    else:
        result = 'need to name'
    return result


def before_point_price(df, point):
    if len(df) >= point:
        open = df['open'][-1 - point]
        close = df['close'][-1 - point]
        low = df['low'][-1 - point]
        high = df['high'][-1 - point]
        volume = df['volume'][-1 - point]
        result = {'open': open, 'close': close, 'low': low, 'high': high, 'volume': volume}
    else:
        print(str(point) + "는 최대 " + str(len(df)) + "을 넘을 수 없습니다.")
        result = 'error'
    return result


def price_criteria_boxplot(df, point):
    bdf = before_point_price(df, point)
    value = bdf['close'] - bdf['open']
    if value >= 0:
        type = "red"
        diff_high = bdf['high'] - bdf['close']
        diff_middle = value
        diff_low = bdf['open'] - bdf['low']
    else:
        type = "blue"
        diff_high = bdf['high'] - bdf['close']
        diff_middle = - value
        diff_low = bdf['open'] - bdf['low']
    result = {'type': type, 'high': diff_high, 'middle': diff_middle, 'low': diff_low}
    return result


def buy_senario_1(df, coin_name):
    levels = 0
    dff = df[-7:]
    box = price_criteria_boxplot(dff, 0)
    a = f_avg_price(dff[-1:])['open'].astype(float)
    b = f_avg_price(dff[-3:])['open'].astype(float)
    c = f_avg_price(dff[-5:])['open'].astype(float)
    d = f_avg_price(dff[-7:])['open'].astype(float)
    if a > b:
        criteria = 'up'
    elif a < b:
        criteria = 'down'
    else:
        if a > c:
            criteria = 'up'
        elif a < c:
            criteria = 'down'
        else:
            if a > d:
                criteria = 'up'
            elif a < d:
                criteria = 'down'
            else:
                criteria = 'normal'
                # 시나리오 1. 상승장 & 주가 상승을 예고하는 캔들 활용

    if bool(box['high'] == 0):
        if criteria == 'up':  # 상승장
            if bool(box['middle'] < box['low']) & bool(box['type'] == 'red'):  # 망치형
                levels = -1
            elif bool(box['type'] == 'red') & bool(box['middle'] == 0) & bool(box['low'] == 0) & bool(
                    box['high'] == 0):
                levels = -1
            elif bool(box['low'] > box['middle'] * 1.5) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] == 0):
                levels = -1
            elif bool(type == 'red') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = -1
            elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 10):
                levels = -1
            elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = -1

            elif box['middle'] > box['low']:  # up
                if box['type'] == 'red':
                    if box['low'] == 0:
                        levels = 1
                    else:
                        levels = 0
                else:
                    if box['low'] == 0:
                        levels = 1
                    else:
                        levels = 0
        elif criteria == 'down':  # 하락장
            if bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 10):
                levels = 1
            elif bool(box['type'] == 'blue') & bool(box['middle'] == 0) & bool(box['low'] == 0) & bool(
                    box['high'] == 0):
                levels = 0
            elif bool(box['low'] > box['middle'] * 1.5) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] == 0):
                levels = 0
            elif bool(type == 'red') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = 0
            elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                    box['middle'] > 0):
                levels = 1
            elif box['middle'] < box['low']:
                if box['type'] == 'red':  # 망치형
                    levels = 1
                else:  # 교수형
                    levels = 0
    return levels


def f_avg_price(df):
    open = round(np.mean(df['open']), -1)
    close = round(np.mean(df['close']), -1)
    low = round(np.mean(df['low']), -1)
    high = round(np.mean(df['high']), -1)
    volume = round(np.mean(df['volume']), -1)
    result = {'open': open, 'close': close, 'low': low, 'high': high, 'volume': volume}
    return result


def box_information(df):
    default_res = []
    for i in range(len(df)):
        df_1 = df[i:i + 1]
        if i == 0:
            default_res = box_create(df_1)
        else:
            next_res = box_create(df_1)
            default_res = pd.concat([default_res, next_res], axis=0)
            default_res.reset_index(drop=True, inplace=True)
    result = box_vervus(default_res)
    return result


def f_rsi(df):
    df = df[-30:]
    a = df.iloc[:, 3] - df.iloc[:, 0]
    b = np.sum(a[a >= 0])
    c = abs(np.sum(a[a < 0]))
    if (b == 0) & (c == 0):
        rsi = 50
    else:
        rsi = (b / (c + b) * 100)
    return rsi


def f_macd(df, a, b, c):
    # a = 12
    # b = 26
    # c = 9
    emaa = f_ema(df, a)[-20:]
    emab = f_ema(df, b)[-20:]
    macd = [(i - j) / j for i, j in zip(emaa, emab)][-20:]
    signal = f_macd_signal(macd, c)[-20:]
    oscillator = [i - j for i, j in zip(macd, signal)]
    result = pd.DataFrame([a for a in zip(oscillator, macd, signal)], columns=['oscillator', 'macd', 'signal'],
                          index=range(len(oscillator)))
    return result


def f_reg_coef(df, name):
    # name = 'open'
    df = df[-5:].reset_index(drop=True)
    y_bol_1 = df[name]
    x_arr = []
    y_arr_bol_1 = []
    for i in range(len(df)):  # i : row
        res = [i + 1]
        x_arr.append(res)
        res = y_bol_1[i:i + 1].to_list()
        y_arr_bol_1.append(res)

    reg = linear_model.LinearRegression()
    reg.fit(x_arr, y_arr_bol_1)
    val_1 = np.sum(reg.coef_.astype(float))
    return val_1


def f_ema(df, length):
    # df = df2
    # length = 12
    sdf = df[0:length].reset_index(drop=True)
    ema = round(np.mean(df.close[0:length]), 0)
    n = np.count_nonzero(df.close.to_list())
    sdf = df[length:n - 1]
    res = [ema]
    ls = sdf.close.to_list()
    for i in range(np.count_nonzero(ls) - 1):
        ema = round(ls[i + 1] * 2 / (length + 1) + ema * (1 - 2 / (length + 1)), 2)
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


def f_oscillator(df):
    oscillator = f_macd(df, 12, 26, 9)  ################# check1
    lines_20 = f_ema(df, 20)[-1:][0]
    lines_10 = f_ema(df, 10)[-1:][0]
    lines_5 = f_ema(df, 5)[-1:][0]
    min_oscillator = np.min(oscillator.oscillator)
    max_oscillator = np.max(oscillator.oscillator)
    coef_oscillator = f_reg_coef(oscillator[-5:].reset_index(drop=True), 'oscillator')
    coef_macd = f_reg_coef(oscillator[-5:].reset_index(drop=True), 'macd')
    coef_signal = f_reg_coef(oscillator[-5:].reset_index(drop=True), 'signal')
    oscillator = oscillator[-1:].reset_index(drop=True).oscillator[0]
    return oscillator, coef_macd, coef_signal, coef_oscillator, min_oscillator, max_oscillator, lines_5, lines_10, lines_20


def check_buy_case2(bdf, min, max):
    df = bdf
    df = bdf[0:len(df) - 1]
    df = df.reset_index(drop=True)
    m = np.max(df.open[0])
    rate = (df.close[4] - m) / m
    criteria_rate = (rate <= -0.005)
    result = False
    temp1 = df[-1:].reset_index(drop=True)
    temp2 = df[-2:].reset_index(drop=True)
    temp3 = df[-3:].reset_index(drop=True)
    temp4 = df[-4:].reset_index(drop=True)
    temp5 = df[-5:].reset_index(drop=True)

    # 3분봉 분석
    criteria0_1 = (temp1.color[
                       0] == 'red')  # & (temp2.color[0] == 'red') & (temp3.color[0] == 'blue') & (temp4.color[0] == 'blue') & (temp5.color[0] == 'blue')
    criteria0_2 = (temp2.rate_volume[0] > 3) & (temp2.rate_volume[0] > 3)
    criteria0 = criteria0_1 & criteria0_2

    # case1 : 지속적 하락 & 값에 큰 변화 없음
    criteria1_1 = np.min(temp1.high[0] - temp1.low[0]) / temp1.low[0]
    criteria1_2 = np.min(temp2.high[0] - temp2.low[0]) / temp2.low[0]
    criteria1_3 = np.min(temp3.high[0] - temp3.low[0]) / temp3.low[0]
    criteria1_4 = np.min(temp4.high[0] - temp4.low[0]) / temp4.low[0]
    criteria1_5 = np.min(temp5.high[0] - temp5.low[0]) / temp5.low[0]
    criteria1 = (criteria1_1 <= criteria1_2 <= criteria1_3 <= criteria1_4 <= criteria1_5)

    # case2 : 큰상승 세력 vs 강력 저항선
    criteria2_1 = (temp1.high[0] - temp1.open[0] == 0) & (temp1.low[0] < np.min(df.low)) & (temp1.color[0] == 'blue')
    criteria2_2 = (temp2.high[0] < max) & ((temp2.open[0] - temp2.low[0]) / temp2.low[0] < abs(0.001)) & (
                temp2.color[0] == 'red')
    criteria2 = criteria2_1 & criteria2_2 & (np.sum(df[-5:].color == "red") >= 2) & (
                np.sum(df[-5:].color == "blue") >= 2)

    # case3 : 강력한 하락 & 반등
    criteria3_1 = ((temp1.open[0] - temp1.close[0]) >= (temp1.close[0] - temp1.low[0]) * 2) * (temp2.color[0] == 'blue')
    criteria3_2 = temp1.high[0] >= temp1.high[0]
    criteria3_3 = np.sum(df[-4:].color == 'blue') == 4
    criteria3_4 = (temp1.high[0] - temp1.open[0]) * 1.3 >= (temp1.close[0] - temp1.low[0])
    criteria3 = criteria3_1 & criteria3_2 & criteria3_3 & criteria3_4

    # case4 : 강력한 하락 & 잠시 반등 - 2%
    criteria4_1 = (np.max(df[-5:].high) > max * 0.95) & (np.min(df[-5:].open) <= min)
    criteria4_2 = ((temp1.close[0] - temp1.low[0]) / temp1.low[0] >= 0.005) & (
                (temp1.open[0] - temp1.close[0]) / temp1.close[0] >= 0.005) & (temp1.high[0] <= temp2.close[0])
    criteria4 = criteria4_1 & criteria4_2

    # case5 : 반등후 하락 & 다시 상승
    criteria5_1 = (temp1.high[0] >= temp2.high[0]) & (
                (temp1.high[0] - temp1.open[0]) >= (temp1.close[0] - temp1.low[0]) * 1.5) & (
                              (temp1.open[0] - temp1.close[0]) * 1.5 <= (temp1.close[0] - temp1.low[0]))
    criteria5_2 = ((temp2.high[0] - temp2.open[0]) >= (temp2.open[0] - temp2.close[0]) * 2) & (
                temp2.low[0] > temp1.close[0])
    criteria5_3 = (np.sum(df[-4:].color == 'blue') == 4) & (temp5.color[0] == 'red') & (
                (temp5.open[0] - temp5.close[0]) / temp5.close[0] <= 0.0001)
    criteria5 = criteria5_1 & criteria5_2 & criteria5_3

    # case6 : 하락증가 & 반등
    criteria6_1 = (temp1.high[0] > temp2.high[0]) & (temp2.close[0] == temp1.open[0]) & (
                temp1.close[0] <= temp2.low[0]) & (
                              (temp1.close[0] - temp1.low[0]) * 1.5 <= (temp1.high[0] - temp1.open[0]))
    criteria6_2 = (temp3.open[0] == temp3.high[0]) & (temp3.color[0] == 'blue')
    criteria6_3 = (temp4.open[0] == temp4.close[0]) & (temp4.color[0] == 'blue')
    criteria6_4 = (temp5.open[0] == temp5.close[0]) & (temp5.color[0] == 'blue')
    criteria6 = criteria6_1 & criteria6_2 & criteria6_3 & criteria6_4

    # case7 : 점핑 하락
    criteria7_1 = (temp1.open[0] == temp2.low[0]) & (temp1.open[0] == temp1.close[0])
    criteria7_2 = (temp2.open[0] == temp2.high[0]) & (
                (temp2.open[0] - temp2.close[0]) * 1.5 <= (temp2.close[0] - temp2.low[0]))
    criteria7_3 = np.sum(df[-5:].color == "blue") >= 4
    criteria7 = criteria7_1 & criteria7_2 & criteria7_3
    if criteria_rate & criteria0 & (criteria1 | criteria2 | criteria3 | criteria4 | criteria5 | criteria6 | criteria7):
        result = True
    return result


def search_coin(coin_name, interval):
    # coin_name = 'KRW-ETH'
    # interval = 'minute1'
    buy_point = 0
    sell_point = 0
    df = []
    while len(df) == 0:
        df = pyupbit.get_ohlcv(coin_name, interval=interval, count=50)
        time.sleep(0.1)
    validation = coin_information(df, coin_name, interval)
    df2 = df[-7:]
    rsi = f_rsi(df)
    oscillator, coef_macd, coef_signal, coef_oscillator, min_oscillator, max_oscillator, lines_5, lines_10, lines_20 = f_oscillator(
        df)
    levels = buy_senario_1(df2, coin_name)

    buy_point = np.sum(validation.get("up_levels") > validation.get("down_levels")) \
                + np.sum(rsi <= 28) \
                + np.sum(np.sum([(coef_macd > 0), (coef_signal > 0), (coef_oscillator > 0)]) >= 2) \
                + np.sum(oscillator < min_oscillator) \
                + np.sum(validation.get('check_buy_2')) \
                + np.sum(validation.get('check_buy'))\
                + np.sum(validation.get("up_levels") > validation.get("down_levels"))

    sell_point = np.sum(validation.get("up_levels") < validation.get("down_levels")) \
                 + np.sum(rsi >= 72) \
                 + np.sum(np.sum([(coef_macd < 0), (coef_signal < 0), (coef_oscillator < 0)]) >= 3) \
                 + np.sum(oscillator > max_oscillator) \
                 + np.sum(validation.get("up_levels") < validation.get("down_levels"))

    min_vs = np.mean(df[-20:].close) - np.std(df[-20:].close)
    if validation.get("min_value") > min_vs:
        # min_value = np.mean([validation.get("min_value"), np.min(df[-20:].close)])
        min_value = validation.get("min_value")
    else:
        min_value = min_vs

    max_vs = np.mean(df[-20:].close) + np.std(df[-20:].close)
    if validation.get("max_value") <= max_vs:
        max_value = validation.get("max_value")
    else:
        max_value = max_vs

    result = {'coin_name': coin_name, 'buy_point': buy_point, 'sell_point': sell_point,
              'influence':validation.get('influence_v'), 'over_volume':validation.get('over_volume'),
              'levels':levels, 'up_levels':np.sum(validation.get("up_levels")),'down_levels':np.sum(validation.get("up_levels")),
              'max':max_value,'min':min_value, 'currency':validation.get("price_currency"),
              'rsi':rsi, 'coef_macd':coef_macd,'coef_signal':coef_signal,'coef_oscillator':coef_oscillator,
              'vs_oscillator': np.sum(oscillator < min_oscillator),'lines_5':lines_5, 'lines_10':lines_10, 'lines_20':lines_20,
              'check_buy': validation.get('check_buy'), 'check_buy_2': validation.get('check_buy_2')
              }
    return result


def get_currency(coin_name):
    cur_price = []
    while len(cur_price) == 0:
        try:
            cur_price = [pyupbit.get_current_price(coin_name)]
        except:
            print(coin_name + "의 현재가 조회 실패")
        time.sleep(0.1)
    cur_price = cur_price[0]
    return cur_price


def coin_information(df, coin_name, interval):
    # coin_name = tickers[1]
    price_currency = get_currency(coin_name)
    # df = df[-7:]
    # price_changed_ratio = changed_ratio(df, price_currency) # changed_ratio
    box = box_information(df)  ### criteria_boxplot

    # start_date_ask = datetime.now()
    # end_date_ask = period_end_date(interval)
    # start_date_bid = end_date_ask + timedelta(seconds=1)
    # end_date_bid = start_date_bid + (end_date_ask - start_date_ask)
    # box['start_date_ask'] = start_date_ask
    # box['end_date_ask'] = end_date_ask
    # box['start_date_bid'] = start_date_bid
    # box['end_date_bid'] = end_date_bid
    # box['price_changed_ratio'] = price_changed_ratio
    box['interval'] = interval
    box['coin_name'] = coin_name
    box['price_currency'] = price_currency
    result = box
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


def changed_ratio(df, price_currency):
    if len(df) > 0:
        df_1 = df[-2:].reset_index(drop=True)
        changed_ratio0 = (df_1.close[0] - df_1.open[0]) / df_1.open[0]
        changed_ratio1 = (df_1.close[1] - df_1.open[1]) / df_1.open[1]
        result = changed_ratio0 - changed_ratio1
    return result


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


# 호가를 이용
def get_hoga_price(coin_name, price, type):
    # input : coint_name
    # price = bid_price
    orderbook = []
    while len(orderbook) == 0:
        orderbook = pyupbit.get_orderbook(coin_name)
    df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
    power_buy = np.sum(df_orderbook.ask_size) >= np.sum(df_orderbook.bid_size) * 3
    power_sell = np.sum(df_orderbook.ask_size) * 3 <= np.sum(df_orderbook.bid_size)
    if type == 'buy':  # 구매가
        bid_price = df_orderbook.bid_price
        price = 80
        idx = np.where(bid_price < price)
        try:
            no = np.min(idx)
            result = bid_price[no]
        except:
            result = coin_buy_price(coin_name)

    elif type == 'bid':  # 구매가
        ask_price = df_orderbook.ask_price
        idx = np.where(ask_price > price)
        try:
            no = np.min(idx)
            result = ask_price[no]
        except:
            result = coin_sell_price(coin_name)
    else:
        result = 0
    return result, power_buy, power_sell


# 현재 내 코인 정보
def my_coin(upbit, remove_tickers):
    my_balances = pd.DataFrame(upbit.get_balances())
    if len(my_balances) < 2:
        print('판매 가능한 코인이 없습니다.')
    else:
        my_balances['coin_name'] = my_balances.unit_currency + '-' + my_balances.currency
        for remove_coine in remove_tickers:
            my_balances = my_balances[my_balances.currency != remove_coine]
        my_balances.reset_index(drop=True, inplace=True)
        my_balances = my_balances[pd.to_numeric(my_balances.balance) > 0]
        my_balances = my_balances[pd.to_numeric(my_balances.avg_buy_price) > 0]
    return my_balances


def coin_buy_price(coin_name):
    # coin_name = 'KRW-BTC'
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
        x = np.max([np.min(check) - 1, 0])
        no = np.min([x, 14])
    price = df_orderbook.bid_price[no]
    return price


def buy_coin(upbit, coin_name, ask_price, last_investment):
    # coin_name = 'KRW-KMD'
    hoga_price, power_buy, power_sell = get_hoga_price(coin_name, ask_price, "buy")
    price = np.min([hoga_price, round_price(ask_price)])
    ask_list = []
    count = last_investment / price
    print('상승 예측 / 매수 요청: ' + coin_name + "    power_buy :" + str(power_buy) + "    hoga :" + str(
        price) + "    count:" + str(price * count))
    try:
        result = upbit.buy_limit_order(coin_name, price, count)
        print(result)
        time.sleep(0.5)
        if len(result) > 2:
            uuid = result.get('uuid')
            ask_list.append(uuid)
        else:
            pass
            # print("service error")
    except:
        print('매수 error: ' + coin_name)

    return ask_list


def bid_coin(upbit, coin_name, buy_interval, buy_point, my_balances, cutoff, benefit, investment):
    # coin_name = 'KRW-SBD'
    res = search_coin(coin_name, buy_interval)
    max = res.get('max_value')
    df = my_balances[my_balances.coin_name == coin_name]
    df.reset_index(drop=True, inplace=True)
    avg_price = pd.to_numeric(df.avg_buy_price)[0]
    hoga_price, power_buy, power_sell = get_hoga_price(coin_name, np.max([avg_price * (1 + benefit), max]), "bid")
    # hoga_price, power_buy, power_sell = get_hoga_price(coin_name, max, "bid")
    current_price = pyupbit.get_current_price(coin_name)
    ratio = (current_price - avg_price) / avg_price
    balance = float(df.balance[0])
    bid_list = []
    if avg_price * balance < 50000:
        benefit = 0.05
        cutoff = 0.03
    elif avg_price * balance < 100000:
        benefit = 0.04
        cutoff = 0.025
    elif avg_price * balance < 150000:
        benefit = 0.03
        cutoff = 0.02
    elif avg_price * balance < 200000:
        benefit = 0.02
        cutoff = 0.015
    elif avg_price * balance >= 200000:
        benefit = 0.01
        cutoff = 0.01

    type2 = "없음"
    print('sell_point: ' + str(res.get('sell_point')) + '    ratio: ' + str(ratio))
    if (0 > ratio > (-abs(cutoff))) & (res.get('buy_point') >= 3) & (res.get('sell_point')==0):
        type2 = "추가매수"
        interval = buy_interval
        res = search_coin(coin_name, interval)
        print(res)
        # b = res.get('buy_point') >= 2
        # coin_name ='KRW-MVL'
        df = pd.DataFrame(res, index=[0])
        criteria = (df.buy_point >= 1) & (df.up_levels >= 1) & (df.down_levels >= 1) & (df.up_levels >=df.down_levels) & (
                df.buy_point > df.sell_point) & (df.coef_oscillator > 0) & (df.rsi < 70)
        df = df[criteria]
        if len(df)>0:
            ask_price = round_price(res.get('min_value'))
            last_investment = round(investment * np.max([res.get('buy_point'), 1]), -3)
            ask_result = buy_coin(upbit, coin_name, ask_price, last_investment)
        if len(ask_result) == 0:
            pass
        else:
            bid_list.append(ask_result)
            # print("구매 탐색중입니다.")

    elif (ratio < (-abs(cutoff))):  # | power_sell: # 손절
        type2 = "손절"
        try:
            result = upbit.sell_limit_order(coin_name, current_price, balance)
            print("판매 요청: " + coin_name + " / 수익률: " + str(round(ratio, 2)))
            if len(result) > 2:
                uuid = result.get('uuid')
                bid_list.append(uuid)
            else:
                pass
                # print("service error")
        except:
            print("service error : 손절 시도")
    elif (ratio > benefit) & (balance * avg_price >= 5500) & (res.get('sell_point') >= 1):
        type2 = "분할매도"
        balance_n = balance
        print("코인 매도 요청: " + coin_name)
        hoga_price = [hoga_price, round_price(max)]
        for price in hoga_price:
            try:
                balance_n = balance * 0.8
                if balance_n * price < 5500:
                    balance_n = balance
                balance = balance - balance_n
                result = upbit.sell_limit_order(coin_name, price, balance_n)
                print(result)
                print(balance_n * price)
                if len(result) > 2:
                    uuid = result.get('uuid')
                    bid_list.append(uuid)
            except:
                print("매도 error")
    elif (ratio > 0.02) & (balance * avg_price < 5500):
        type2 = "2만원 미만 처분"
        try:
            upbit.buy_market_order(coin_name, 6000)
            time.sleep(0.5)
            upbit.sell_market_order(coin_name, balance)
        except:
            print("매도 error")
    print(str(datetime.now()) + "coin_name: " + coin_name + "    매도유형:" + str(type2) + "    ratio: " + str(ratio))
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
            print('현재가 조회 실패: ' + coin_name)
    KRW = pd.to_numeric(upbit.get_balances()[0].get('balance'))
    my_balances['tot_price'] = pd.to_numeric(my_balances['balance']) * pd.to_numeric(my_balances['avg_buy_price'])
    my_balances['currecy_price'] = pd.to_numeric(currency_price)
    my_balances['tot_currecy_price'] = pd.to_numeric(my_balances['balance']) * pd.to_numeric(currency_price)

    my_balances['pnl'] = (pd.to_numeric(my_balances['currecy_price']) - pd.to_numeric(
        my_balances['avg_buy_price'])) * pd.to_numeric(my_balances['balance'])
    # coin_validation = my_balances.sort_values('pnl', ascending= False)
    a = np.sum(my_balances['tot_price'])
    b = np.sum(my_balances['tot_currecy_price'])
    my_money_value = KRW + b
    revenue = (b - a)
    revenue_ratio = revenue / a
    result = {'my_money_value': round(my_money_value, 2), 'KRW': round(KRW, 0), 'my_revenue': round(revenue, 0),
              'my_revenue_ratio': round(revenue_ratio, 2), 'tot_price': round(a, 0), 'tot_currency_price': round(b, 0)}
    return result


def merge_df(df, directory, name):
    # df = result
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
    if len(df) > 0:
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
        df.to_json(json_file, orient='table')
    else:
        try:
            os.remove(json_file)
        except:
            print("제거error: " + json_file)
    return df


def get_coin_selection(access_key, secret_key, search_interval):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        start_time = time.time()
        my_balances = []
        tickers = pyupbit.get_tickers(fiat="KRW")
        for coin_name in tickers:
            # coin_name = tickers[0]
            res = search_coin(coin_name, search_interval)
            my_balances.append(res)
            time.sleep(0.1)
        df = pd.DataFrame(my_balances).sort_values('buy_point', ascending=False)
        rate = [np.sum(pd.DataFrame(my_balances).buy_point > 1) / len(pd.DataFrame(my_balances).buy_point)]
        criteria = (df.buy_point >= 1) & (df.up_levels>=1) & (df.down_levels>=1) & (df.buy_point > df.sell_point) & (df.coef_oscillator > 0) & (df.rsi < 70)
        df = df[criteria].reset_index(drop=True)
        print('***************후보 산출 정보***************')
        print(df)
        # df = df[(df.volume_levels >= 2) & (df.updown_levels >= 2)].sort_values('volume_levels', ascending=False).reset_index(drop=True)
        directory = 'buy_selection_coin'
        try:
            os.remove('buy_selection_coin/buy_selection_coin.json')
        except:
            pass
        if len(df) >= 1:
            merge_df(df, 'buy_selection_coin', 'buy_selection_coin.json')
            print(str(datetime.now()) + "    구매 후보로 선정된 코인: " + ', '.join(df.coin_name))
        else:
            print(str(datetime.now()) + "    구매 후보로 선정된 코인: 없음")
        end_time = time.time()
        diff_time = end_time - start_time
        print("후보 산출시간: " + str(diff_time) + "초")
        time.sleep(1)
    return result


def f_my_coin(upbit):
    df = pd.DataFrame(upbit.get_balances())
    time.sleep(0.1)
    df.reset_index(drop=True, inplace=True)
    df['coin_name'] = df.unit_currency + '-' + df.currency
    df['buy_price'] = pd.to_numeric(df.balance, errors='coerce') * pd.to_numeric(df.avg_buy_price, errors='coerce')
    df = df[df.buy_price>0]
    df.reset_index(drop=True, inplace=True)
    return df


def round_price(price):
    if price < 10:
        price = round(price, 2)
    elif price < 100:
        price = round(price, 1)
    elif price < 1000:
        price = round(price / 5, 0) * 5
    elif price < 10000:
        price = round(price / 10, 0) * 10
    elif price < 100000:
        price = round(price / 50, 0) * 50
    elif price < 1000000:
        price = round(price / 500, 0) * 500
    else:
        price = round(price / 1000, 0) * 1000
    return price


def execute_sell_schedule(access_key, secret_key, buy_interval, buy_point, cutoff, benefit, investment, sec):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        my_balances = []
        while len(my_balances) == 0:
            my_balances = f_my_coin(upbit)
            time.sleep(0.1)
        tickers = (my_balances.coin_name).to_list()
        res = []
        bid_list = []
        print("판매 코인: " + str(tickers))
        for coin_name in tickers:
            try:
                # coin_name = tickers[0]
                # check = check_sell_combo(coin_name)

                bid_result = []
                bid_result = bid_coin(upbit, coin_name, buy_interval, buy_point,  my_balances, cutoff, benefit, investment)

                if len(bid_result) == 0:
                    pass
                else:
                    bid_list.append(bid_result)
                # except:
                #     print('매도 error: ' + coin_name)
                time.sleep(0.5)
            except:
                pass
        time.sleep(sec)
        reservation_cancel(upbit, bid_list)

def check_sell_combo(coin_name):
    check = False
    res = search_coin(coin_name, "minute60")
    a = res.get('sell_point') >= 1
    if a:
        print(res)
        check = True
        # res = search_coin(coin_name, "minute30")
        # a = res.get('sell_point') >= 1
        # if a:
        #     res = search_coin(coin_name, "minute15")
        #     a = res.get('sell_point') >= 1
        #     if a:
        #         res = search_coin(coin_name, "minute5")
        #         a = res.get('sell_point') >= 1
        #         if a:
        #             res = search_coin(coin_name, "minute5")
        #             a = res.get('sell_point') >= 1
        #             if a:
        #                 check = True
    return check


def execute_buy_schedule(access_key, secret_key, buy_interval, buy_point, investment, sec):
    upbit = pyupbit.Upbit(access_key, secret_key)
    while True:
        money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
        while money > 5000:
            reserv_list = []
            load = []
            rate = []
            while (len(load) == 0):
                load = load_df('buy_selection_coin', 'buy_selection_coin.json')
            load = load.sort_values('buy_point', ascending=False).reset_index()
            tickers = list(set(load.coin_name))
            if investment < 10000:
                investment = money
            # load = load.sort_values('buy_point', ascending=True).reset_index()
            # tickers = list(set(load.coin_name))
            time.sleep(1)
            #

            # else:
            # 구매
            print('*************************************************')
            print('********************* 매수 시작 *******************')
            if len(tickers)==0:
                tickers = pyupbit.get_tickers(fiat="KRW")
            ask_list = []
            for coin_name in tickers:
                #coin_name = tickers[0]
                try:
                    res = search_coin(coin_name, buy_interval)
                        # b = res.get('buy_point') >= 2
                        # coin_name ='KRW-MVL'
                    print(res)
                    ask_price = round_price(res.get('min'))
                    last_investment = round(investment * np.max([res.get('buy_point'), 1]), -3)
                    print("last_investment: " + str(last_investment) + "   ask_price: " + str(ask_price))
                    df = pd.DataFrame(res, index=[0])
                    criteria = (df.buy_point >= 1) & (df.up_levels >= 1) & (df.down_levels >= 1) & (
                                df.up_levels >= df.down_levels) & (
                                       df.buy_point > df.sell_point) & (df.coef_oscillator > 0) & (df.rsi < 70)
                    df = df[criteria]
                    if len(df)>0:
                        ask_result = buy_coin(upbit, coin_name, ask_price, last_investment)
                    if len(ask_result) == 0:
                        pass
                    else:
                        ask_list.append(ask_result)
                        # print("구매 탐색중입니다.")
                    money = float(pd.DataFrame(upbit.get_balances())['balance'][0])
                    ask_list = list(set(ask_list))
                    if money < 5000:
                        time.sleep(sec / 2)
                        reservation_cancel(upbit, ask_list)
                except:
                    print("매수 error")

            print('******************')
            print("*매수 예약 취소 대기*")
            print('******************')
            time.sleep(sec)
            reservation_cancel(upbit, ask_list)


def load_df(directory, name):
    json_file = directory + '/' + name
    ori_df = []
    if os.path.exists(json_file):
        ori_df = pd.read_json(json_file, orient='table')
    return ori_df


def reservation_cancel(upbit, reserv_list):
    if len(reserv_list) > 0:
        for uuids in reserv_list:
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