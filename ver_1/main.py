import buying_module2 as co
from multiprocessing import Process, Queue

def coin_trade(access_key, secret_key, buy_interval, buy_point, search_interval, investment, cutoff, benefit, sec):
    th3 = Process(target=co.get_coin_selection, args=(access_key, secret_key, search_interval))
    th1 = Process(target=co.execute_buy_schedule, args=(access_key, secret_key, buy_interval, buy_point, investment, sec))
    th2 = Process(target=co.execute_sell_schedule, args=(access_key, secret_key, buy_interval, buy_point, cutoff, benefit, investment, sec))
    result = Queue()
    th1.start()
    th2.start()
    th3.start()
    th1.join()
    th2.join()
    th3.join()

if __name__ == '__main__':
    ##################################
    # input
    ##################################
    access_key = '5RsZuqMZ6T0tfyjNbIsNlKQ8LI4IVwLaYMBXiaa2'  # ''
    secret_key = 'zPKA1zJwymHMvUSQ2SqYWDgkxNgVfG7Z5jiNLcaJ'  # ''
    investment = 10000
    buy_interval = 'minute10'
    buy_point = 2
    search_interval = 'minute60'
    #tickers = [] # 예시 : tickers = ['KRW-BTC','KRW-ETC']
    cutoff = 0.03 # 1% 손절
    benefit = 0.008  # 1% 익절
    sec = 10
    result = coin_trade(access_key, secret_key, buy_interval, buy_point, search_interval, investment, cutoff, benefit, sec)
