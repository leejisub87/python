import buying_module as co

#접속
intervals = ["minute1", "minute3", "minute5", "minute10", "month", "days", "minute240", "minute60", "minute30", "minute15"]

##################################
# input
##################################
access_key = '0PUcnCxxr9pax8xExCpLvMndcIpMcY38P9Q3cfRH'
secret_key = 'xxACrqJLnXj0JH9DLP0839RWnpN2Izrne6mZZoMx'
interval = "minute3"
remove_tickers = ['KRW', 'XYM']
investment = 10000
cutoff = 0.01  # 1% 손절
benefit = 0.02  # 1% 익절

result = co.main(access_key, secret_key, interval, remove_tickers, investment, cutoff, benefit)