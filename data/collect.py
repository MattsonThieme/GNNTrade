import datetime
import time
import ccxt
import csv

exchange = ccxt.binance ()

assets = ['BTC', 'ETH', 'XLM', 'CVC']
symbols = [i+'/USDT' for i in assets]

data = exchange.fetch_tickers(symbols)
params = ['ask']

target = 5  # delay target

with open('{}_{}s.csv'.format("-".join(assets), target), 'w') as r:
    reader = csv.reader(r)
    try:
        row1 = next(reader)
    except:
        writer = csv.writer(r)
        headers = [i+"-"+j for i in assets for j in params]
        writer.writerow(headers)


while True:

    try:
        start = time.time()
        data = exchange.fetch_tickers(symbols)
        vals = [data[i][j] for i in symbols for j in params]
        with open('{}_{}s.csv'.format("-".join(assets), target), "a") as r:
            reader = csv.reader(r)
            try:
                row1 = next(reader)
            except:
                writer = csv.writer(r)
                writer.writerow(vals)

        end = time.time()

        time.sleep(target - (end-start))

    except:
        time.sleep(target)
