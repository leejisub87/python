import numpy as np
import pandas as pd
import pyupbit
import time
from datetime import datetime, timedelta
import os
import math
#from multiprocessing import Process, Queue

pd.set_option('display.max_columns', 20)

class search_coin:
    def __init__(self, coin_name, interval):
        self.coin_name = coin_name
        self.interval = interval
        self.df = pyupbit.get_ohlcv(self.coin_name, interval=self.interval, count=10)
    def search(self):
        self.df = pyupbit.get_ohlcv(self.coin_name, interval=self.interval, count=7)
    def excute(self):
        buy_point = 0
        sell_point = 0
        levels = self.buy_senario_1()
        if levels > 0: # 구매
            buy_point = levels
        elif levels <0:# 판매
            sell_point = abs(levels)

        result_2 = self.buy_senario_2()
        if result_2.get('check_buy'):
            buy_point = buy_point + 1 + result_2.get('influence_v')
        elif result_2.get('check_sell'):
            sell_point = sell_point + 1 + result_2.get('influence_v')
        result = {'buy_point': buy_point, 'sell_point':sell_point }
        return result
    ############################
    #### senario1 - 시작
    ############################
    def buy_senario_1(self):
        coin_name = self.coin_name

        df = self.df
        a = float(self.avg_price(df[-1:])['open'])
        b = float(self.avg_price(df[-3:])['open'])
        c = float(self.avg_price(df[-5:])['open'])
        d = float(self.avg_price(df[-7:])['open'])

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
        levels = 0
        box = self.price_criteria_boxplot(0)
        if bool(box['high'] == 0):
            if criteria == 'up':  # 상승장
                if bool(box['middle'] < box['low']) & bool(box['type'] == 'red'):  # 망치형
                    levels = -1
                    print(coin_name + '의 주가 (소)하락을 예고')
                elif bool(box['type'] == 'red') & bool(box['middle'] == 0) & bool(box['low'] == 0) & bool(
                        box['high'] == 0):
                    levels = 0
                elif bool(box['low'] > box['middle'] * 1.5) & bool(box['high'] > box['middle'] * 1.5) & bool(
                        box['middle'] == 0):
                    levels = -1
                    print(coin_name + '의 주가 (소)하락을 예고')
                elif bool(type == 'red') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                        box['middle'] > 0):
                    levels = -1
                    print(coin_name + '의 주가 (소)하락을 예고')
                elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 10):
                    levels = -2
                    print(coin_name + '의 주가 (대)하락을 예고')
                elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                        box['middle'] > 0):
                    levels = -2
                    print(coin_name + '의 주가 (대)하락을 예고')

                elif box['middle'] > box['low']:  # up
                    if box['type'] == 'red':
                        if box['low'] == 0:
                            levels = 2
                            print(coin_name + '의 주가 (대)상승을 예고')
                        else:
                            levels = 1
                            print(coin_name + '의 주가 (소)상승을 예고')
                    else:
                        if box['low'] == 0:
                            levels = 2
                            print(coin_name + '의 주가 (대)상승을 예고')
                        else:
                            levels = 1
                            print(coin_name + '의 주가 (소)상승을 예고')
            elif criteria == 'down':  # 하락장
                if bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 10):
                    levels = 2
                    print(coin_name + '의 주가 (대)상승을 예고')
                elif bool(box['type'] == 'blue') & bool(box['middle'] == 0) & bool(box['low'] == 0) & bool(
                        box['high'] == 0):
                    levels = 0
                elif bool(box['low'] > box['middle'] * 1.5) & bool(box['high'] > box['middle'] * 1.5) & bool(
                        box['middle'] == 0):
                    levels = 1
                    print(coin_name + '의 주가 (소)상승을 예고')
                elif bool(type == 'red') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                        box['middle'] > 0):
                    levels = 1
                    print(coin_name + '의 주가 (소)상승을 예고')
                elif bool(type == 'blue') & bool(box['low'] == 0) & bool(box['high'] > box['middle'] * 1.5) & bool(
                        box['middle'] > 0):
                    levels = 2
                    print(coin_name + '의 주가 (대)반등을 예고')
                elif box['middle'] < box['low']:
                    if box['type'] == 'red':  # 망치형
                        levels = 2
                        print(coin_name + '의 주가 (큰)반등을 예고')
                    else:  # 교수형
                        levels = 1
                        print(coin_name + '의 주가 (소)반등을 예고')

        return levels
    def price_criteria_boxplot(self, point):
        bdf = self.before_point_price(point)
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
    def before_point_price(self, point):
        df = self.df[-7:]
        open = df['open'][-1 - point]
        close = df['close'][-1 - point]
        low = df['low'][-1 - point]
        high = df['high'][-1 - point]
        volume = df['volume'][-1 - point]
        result = {'open': open, 'close': close, 'low': low, 'high': high, 'volume': volume}
        return result
    def avg_price(self, df):
        open = round(np.mean(df['open']), -1)
        close = round(np.mean(df['close']), -1)
        low = round(np.mean(df['low']), -1)
        high = round(np.mean(df['high']), -1)
        volume = round(np.mean(df['volume']), -1)
        result = {'open': open, 'close': close, 'low': low, 'high': high, 'volume': volume}
        return result
    ############################
    #### senario1 - 종료
    ############################
    def buy_senario_2(self):
        return self.box_information()
    def check_buy_case(self,df):
        idx = len(df)
        updown = self.criteria_updown(df)
        status = []
        if bool(np.sum(updown.volume_status == 'down') >= 3) & bool(np.sum(updown.volume_status == 'down') >= 3):
            status = 'down'
        elif bool(np.sum(updown.volume_status == 'up') >= 3) & bool(np.sum(updown.volume_status == 'up') >= 3):
            status = 'up'

        result = False
        if bool(status == 'down') & bool(df.chart_name[4] == 'shooting_star'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[4] == 'upper_tail_pole_yangong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[2] == 'lower_tail_pole_yangong') & bool(
                df.chart_name[3] == 'upper_tail_pole_yangong') & bool(df.chart_name[4] == 'lower_tail_pole_yangong'):
            result = True
        elif bool(df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4] == 'shortbody_yangbong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[3] == 'longbody_umbong') & bool(
                df.chart_name[4] in ['shortbody_yangbong', 'rickshawen_doji', 'longbody_yangbong']):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[3] == 'longbody_umbong') & bool(
                df.chart_name[4] in ['longbody_yangbong', 'rickshawen_doji']):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(
                df.chart_name[3] == 'shortbody_yangbong') & bool(
                df.chart_name[4] == 'longbody_yangbong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(
                df.chart_name[3] == 'doji_cross') & bool(
                df.chart_name[4] == 'longbody_yangbong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(
                df.chart_name[3] == 'longbody_yangbong') & bool(
                df.chart_name[4] == 'shortbody_yangbong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[2] == 'longbody_umbong') & bool(
                df.chart_name[3] == 'longbody_yangbong') & bool(
            df.chart_name[4] == 'shortbody_yangbong'):
            result = True
        elif bool(status == 'down') & bool(np.sum(df.color == 'blue') >= 4) & bool(
                df.chart_name[3] == 'shooting_star') & bool(df.chart_name[4] == 'longbody_yangbong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[3] == 'shooting_star') & bool(
                df.chart_name[4] == 'pole_yangbong'):
            result = True
        elif bool(status == 'down') & bool(df.chart_name[3] == 'longbody_umbong') & bool(
                df.chart_name[4] in ['longbody_yangbong', 'shortbody_yangbong']):
            result = True
        elif bool(np.sum(df.color == 'blue') >= 4) & bool(
                df.chart_name[4] in ['stone_cross', 'lower_tail_pole_umbong']):
            result = True
        elif bool(np.sum(df.color == 'blue') >= 4) & bool(
                df.chart_name[4] in ['dragonfly_cross', 'upper_tail_pole_yangong']):
            result = True
        elif bool(np.sum(df.color == 'blue') >= 4) & bool(np.sum(df.chart_name == 'pole_umbong') >= 2) & bool(
                df.chart_name[4] in ['shooting_star', 'upper_tail_pole_umbong']):
            result = True
        elif bool(np.sum(df.color == 'blue') >= 4) & bool(df.chart_name[4] in ['doji_cross', 'spinning_tops']):
            result = True
        elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong') >= 2) & bool(
                np.sum(df.chart_name == 'longbody_yangbong') >= 1) & bool(df.chart_name[4] in ['longbody_umbong']):
            result = True
        elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong') >= 2) & bool(
                np.sum(df.chart_name == 'shortbody_umbong') >= 1):
            result = True
        elif bool(status == 'down') & bool(np.sum(df.chart_name == 'longbody_umbong') >= 2) & bool(
                np.sum(df.chart_name[4] in ['shooting_star', 'lower_tail_pole_umbong']) >= 1):
            result = True
        return result
    def check_sell_case(self,df):
        idx = len(df)
        updown = self.criteria_updown(df)
        result = False
        if bool(np.sum(updown.volume_status == 'down') >= 3) & bool(np.sum(updown.volume_status == 'down') >= 3):
            status = 'down'
        elif bool(np.sum(updown.volume_status == 'up') >= 3) & bool(np.sum(updown.volume_status == 'up') >= 3):
            status = 'up'
        ####################################################
        #### 하락 반전형 캔들
        ## 유성형
        if bool(status == 'up') & bool(df.chart_name[4] == 'upper_tail_pole_yangong') & bool(df.color[4] == 'blue'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[4] == 'lower_tail_pole_yangbong') & bool(df.color[4] == 'red'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[4] == 'lower_tail_pole_umbong') & bool(df.color[4] == 'red'):
            result = True
        elif bool(status == 'up') & (bool(df.chart_name[3] in ['doji_cross', 'rickshawen_doji']) | bool(
                df.chart_name[4] in ['doji_cross', 'rickshawen_doji'])):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[3] == 'shortbody_yangbong') & bool(
                df.chart_name[4] == 'longbody_yangbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[2] == 'upper_tail_pole_umbong') & bool(
                df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4] == 'lower_tail_pole_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(
                df.chart_name[4] == 'doji_corss'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(
                df.chart_name[4] == 'upper_tail_pole_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(
                df.chart_name[3] == 'longbody_umbong') & bool(df.chart_name[4] == 'shortbody_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(
                df.chart_name[3] == 'shortbody_umbong') & bool(df.chart_name[4] == 'longbody_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(
                df.chart_name[4] == 'longbody_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[3] == 'longbody_yangbong') & bool(
                df.chart_name[4] == 'longbody_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(
                df.chart_name[3] == 'shortbody_yangbong') & bool(df.chart_name[4] == 'longbody_umbong'):
            result = True
        #########################################
        elif bool(status == 'up') & bool(df.chart_name[2] == 'longbody_yangbong') & bool(
                df.chart_name[3] == 'doji_corss') & bool(df.chart_name[4] == 'longbody_pole_umbong'):
            result = True
        elif bool(status == 'up') & bool(df.chart_name[2] == 'shortbody_yangbong') & bool(
                df.chart_name[3] == 'doji_corss') & bool(df.chart_name[4] == 'shortbody_umbong'):
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
    def criteria_updown(self,df):
        res = []
        if len(df) == 1:
            close_rate = 0
            open_rate = 0
            volume_rate = 0
        else:
            for i in range(len(df) - 1):
                a = df[i:i + 2].reset_index(drop=True)
                a.reset_index(drop=True, inplace=True)
                close_rate = df.close[1] / df.close[0]
                open_rate = df.open[1] / df.open[0]
                volume_rate = df.volume[1] / df.volume[0]
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
        t = pd.DataFrame(res).reset_index(drop=True)
        return t
    def coin_predict(self, res):
        df = res[-5:]
        df.reset_index(inplace=True, drop=True)
        default = df[-1:].reset_index()
        v = list(df.rate_volume)
        influence_v = 0
        if bool(v.index(max(v)) in [3, 4]):
            influence_v = 1

        min_buy_price = min(df.open[0], df.close[0]) - 1.96 * np.std(df.length_low[0])
        max_buy_price = min(df.open[0], df.close[0]) + 1.96 * np.std(df.length_low[0])
        max_sell_price = max(df.open[0], df.close[0]) - 1.96 * np.std(df.length_low[0])

        min_sell_price = max(df.open[0], df.close[0]) - 1.96 * np.std(df.length_low[0])
        min_sell_price = max(min_sell_price, min_buy_price)
        remain_price = (min_sell_price - min_buy_price) / min_buy_price

        check_buy = self.check_buy_case(df)
        check_sell = self.check_sell_case(df)
        # 상태
        result = {'remain_price': remain_price, 'influence_v': influence_v, 'check_buy': check_buy,
                  'check_sell': check_sell,
                  'min_buy_price': min_buy_price, 'min_sell_price': min_sell_price,
                  'max_buy_price': max_buy_price, 'max_sell_price': max_sell_price}
        return result
    def period_end_date(self):
        start_time = datetime.now()
        intervals = self.interval
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
    def chart_name(self,df_1):
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
    def box_create(self,df_1):
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
        name = self.chart_name(df_1)
        df_1['chart_name'] = name
        return df_1
    def box_vervus(self,default_res):
        default_df = default_res[-6:]
        v = []
        for i in range(len(default_df) - 1):
            ddf = default_df[i:i + 2].reset_index(drop=True)
            odds_volume = ddf.volume[1] / ddf.volume[0]
            v.append(odds_volume)
        odds_volume = pd.DataFrame(v, columns=['rate_volume'])
        res = pd.concat([default_df[-5:].reset_index(drop=True), odds_volume.reset_index(drop=True)], axis=1)
        result = self.coin_predict(res)
        return result
    def box_information(self):
        df = self.df
        default_res = []
        for i in range(len(df)):
            df_1 = df[i:i + 1]
            if i == 0:
                default_res = self.box_create(df_1)
            else:
                next_res = self.box_create(df_1)
                default_res = pd.concat([default_res, next_res], axis=0)
                default_res.reset_index(drop=True, inplace=True)
        result = self.box_vervus(default_res)
        return result
class coin_sell:
    def __init__(self, upbit):
        self.directory = 'sell_list'
        self.name = 'sell_list.json'
        self.upbit = upbit
        self.my_coin = coin_common().my_coin(upbit)
        self.cutoff = -0.01

    def cancel(self):
        coin_common().reservation_cancel(self.upbit, self.directory, self.name)

    def f_balance(self):
        coin_name = self.coin_name
        my_coin = self.my_coin
        df = self.my_coin[self.my_coin.coin_name == coin_name].reset_index(drop=True)
        avg_price = df['avg_buy_price'].astype(float)
        self.current_price = pyupbit.get_current_price(coin_name)
        self.ratio = (self.current_price - avg_price) / avg_price
        self.balance = df['balance'].astype(float)

    def selling(self, coin_name, point):
        self.coin_name = coin_name
        self.point = point
        self.f_balance()
        self.price()
        self.execute()

    def price(self):
        # input : coint_name
        coin_name = self.coin_name
        orderbook = pyupbit.get_orderbook(coin_name)
        df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
        df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
        df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

        # 매수 > 매도의 가격을 측정
        df_orderbook['selling_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size > x.cum_bid_size, axis='columns')
        check = [i for i, value in enumerate(list(df_orderbook.selling_YN)) if value == False]
        no = max(min(check),self.point)
        self.price = df_orderbook.ask_price[max(no - 1, 0)]

    def execute(self):
        upbit = self.upbit
        coin_name = self.coin_name
        price = self.price
        count = self.balance
        try:
            res = upbit.sell_limit_order(coin_name, price, count)
            time.sleep(0.3)
            if len(res) > 2:
                print("***판매 요청 정보***")
                print("코인: "+coin_name + "/ 가격: " + str(price) + "/ 수량: " + str(count))
                uuid = list()
                uuid.append(res.get('uuid'))
                result = pd.DataFrame(uuid, columns=['uuid'])
                coin_common().merge_df(result, self.directory, self.name)
        except:
            print("판매 error")
class coin_common:
    def merge_df(self, df, directory, name):
        #df = result
        json_file = directory + '/' + name
        # 폴더생성
        if not os.path.exists(directory):
            os.makedirs(directory)
        df = df.dropna(axis=0)
        df = pd.DataFrame(df)
        if os.path.isfile(directory):
            ori_df = pd.read_json(json_file, orient='table')
            df = pd.concat([ori_df, df], axis=0)
            df.reset_index(drop=True, inplace=True)
        else:
            df = df
        df.to_json(json_file, orient='table')
    def load_df(self, directory, name):
        ori_df = []
        json_file = directory + '/' + name
        try:
            name = os.listdir(directory)[-1]
            json_file = directory + '/' + name
            ori_df = pd.read_json(json_file, orient='table')
        except:
            print("데이터 로드 실패: "+json_file)
        return ori_df
    def reservation_cancel(self, upbit, directory, name):
        df = self.load_df(directory, name)
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
    def my_coin(self, upbit):
        my_coin = pd.DataFrame(upbit.get_balances())
        my_coin['coin_name'] = my_coin.unit_currency + '-' + my_coin.currency
        my_coin['buy_price'] = pd.to_numeric(my_coin.balance, errors='coerce') \
                               * pd.to_numeric(my_coin.avg_buy_price, errors='coerce')
        return my_coin
class coin_buy:
    def __init__(self, upbit):
        self.directory = 'buy_list'
        self.name = 'buy_list.json'
        self.upbit = upbit

    def cancel(self):
        coin_common().reservation_cancel(self.upbit,self.directory, self.name)

    def buying(self, coin_name, investment, point):
        self.point = point
        self.coin_name = coin_name
        self.investment = investment
        self.price()
        self.execute()

    def execute(self):
        upbit = self.upbit
        coin_name = self.coin_name
        price = self.price
        count = self.investment / self.price
        count = count * (1 + self.point/10)
        #count = investment / price
        try:
            res = upbit.buy_limit_order(coin_name, price, count)
            time.sleep(0.3)
            if len(res) > 2:
                print("***구매 요청 정보***")
                print("코인: "+coin_name + "/ 가격: " + str(price) + "/ 수량: " + str(count))
                uuid = list()
                uuid.append(res.get('uuid'))
                result = pd.DataFrame(uuid, columns=['uuid']).reset_index(drop=True)
                coin_common().merge_df(result, self.directory, self.name)
        except:
            print("구매 error")

    def price(self):
        coin_name = self.coin_name
        orderbook = pyupbit.get_orderbook(coin_name)
        df_orderbook = pd.DataFrame(orderbook[0]['orderbook_units'])
        df_orderbook['cum_ask_size'] = df_orderbook['ask_size'].apply(lambda x: float(np.cumsum(x)))
        df_orderbook['cum_bid_size'] = df_orderbook['bid_size'].apply(lambda x: float(np.cumsum(x)))

        # 매수 > 매도의 가격을 측정
        df_orderbook['buying_YN'] = df_orderbook.apply(lambda x: x.cum_ask_size < x.cum_bid_size, axis='columns')
        check = [i for i, value in enumerate(list(df_orderbook.buying_YN)) if value == False]
        if len(check)>0:
            no = np.min(check)
        else:
            no = 14
        self.price = df_orderbook.bid_price[max(no - 1, 0)]

def buy_process(upbit, result, coin_name):
    buy_point = result.get('buy_point')
    if buy_point > 0:
        coin_buy(upbit).buying(coin_name, investment, buy_point)  # 구매

def sell_process(cs, result, coin_name):
    sell_point = result.get('sell_point')
    if sell_point > 0:
        KRW = float(cs.my_coin[0:1].balance[0])
        my_buy_price = np.sum(cs.my_coin.buy_price)
        my_trade_coin = cs.my_coin[pd.to_numeric(cs.my_coin.buy_price) > 0]
        tickers = list(my_trade_coin.coin_name)
        cs.selling(coin_name, sell_point)  # 판매

def auto_cancel(upbit):

    coin_sell(upbit).cancel()
    coin_buy(upbit).cancel()

def trade(access_key, secret_key, interval, investment):
    upbit = pyupbit.Upbit(access_key, secret_key)
    tickers = pyupbit.get_tickers(fiat="KRW")
    while True:
        st = time.time()
        diff = 0
        while diff < 20:
            for coin_name in tickers:
                #coin_name = tickers[0]
                result = search_coin(coin_name, interval).excute()
                cs = coin_sell(upbit)
                buy_process(upbit, result, coin_name)
                sell_process(cs, result, coin_name)
                # th1 = Process(target=buy_process, args=(upbit, result, coin_name))
                # th2 = Process(target=sell_process, args=(upbit, cs, result, coin_name))
                #
                # result = Queue()
                # th1.start()
                # th2.start()
                # th1.join()
                # th2.join()
            et = time.time()
            diff = et - st
        auto_cancel(upbit)

if __name__ == '__main__':
    #intervals = ["day", "minute240", "minute60", "minute30", "minute15", 'minute10', 'minute5']
    access_key = 'jVrl599q44MVUdfYznmTmSTTwKOmBx8iIpxHaue1'
    secret_key = 'yoS95G5vZ8upRD2di0SxmSns72DBbVfbsL6kXAT4'
    investment = 10000
    interval = 'minute10'
    trade(access_key, secret_key, interval, investment)

