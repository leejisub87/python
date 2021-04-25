import buying_module as co
# pd.set_option('display.max_columns', 20)

#접속
intervals = ["minute1", "minute3", "minute5", "minute10", "month", "days", "minute240", "minute60", "minute30", "minute15"]
upbit = co.connection_upbit('home')

##################################
# input
##################################
interval = "minute1"
remove_tickers = ['KRW', 'XYM']
investment = 10000
cutoff = 0.01  # 1% 손절
benefit = 0.02  # 1% 익절
ask_no = 1
bid_no = 1

result = co.main(upbit, interval, remove_tickers, investment, cutoff, benefit, ask_no, bid_no)
