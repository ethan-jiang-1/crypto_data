import numpy as np
import pandas as pd
import cryptocompare as cc

# list of coins
coin_list = cc.get_coin_list()
coins = sorted(list(coin_list.keys()))

# get data for all available coins
if __name__ == '__main__':
    coin_data = {}
    for i in range(len(coins)//50 + 1):
        # limited to a list containing at most 300 characters #
        coins_to_get = coins[(50*i):(50*i+50)]
        message = cc.get_price(coins_to_get, curr='USD', full=True)
        coin_data.update(message['RAW'])

    # remove 'USD' level
    for k in coin_data.keys():
        coin_data[k] = coin_data[k]['USD']

    # Now we can go ahead and create a DataFrame  from our coin_data  dictionary and sort it by market capitalization:
    coin_data = pd.DataFrame.from_dict(coin_data, orient='index')
    coin_data = coin_data.sort_values('MKTCAP', ascending=False)

    print(coin_data['MKTCAP'].head())
    # exclude coins that haven't traded in last 24 hours
    # TOTALVOLUME24H is the amount the coin has been traded
    # in 24 hours against ALL its trading pairs
    coin_data = coin_data[coin_data['TOTALVOLUME24H'] != 0]

    import pdb; pdb.set_trace()
    # get the last month’s historical daily data for the 100 top coins by market cap, stored as a dictionary of DataFrames
    top_coins = coin_data[:100].index
    df_dict = {}
    for coin in top_coins:
        hist = cc.get_historical_price_minute(coin, curr='USD', limit=2000)
        if hist:
            hist_df = pd.DataFrame(hist['Data'])
            hist_df['time'] = pd.to_datetime(hist_df['time'], unit='s')
            hist_df.index = hist_df['time']
            del hist_df['time']
            df_dict[coin] = hist_df

    # And we can access the data for any coin in the dictionary by doing
    # df_dict[coin]  where coin is the symbol of the coin we interested in, such
    # as ‘BTC’. Now that we have our data, we can do some fun stuff!

    # pull out closes
    closes = pd.DataFrame()
    for k, v in df_dict.items():
        closes[k] = v['close']
    # re-order by market cap
    closes = closes[coin_data.index[:100]]

    # some cool stuff we can do with our data
    # plot 2017 prices
    import matplotlib.pyplot as plt
    import seaborn as sns
    # plot some prices
    closes.loc['2017', ['BTC', 'ETH', 'LTC']].plot()

    # plot some returns
    closes.loc['2017', ['BTC', 'ETH', 'LTC']].pct_change().plot()

    # plot correlation matrx
    sns.heatmap(closes.loc['2017', ['BTC', 'ETH', 'LTC', 'XRP', 'XUC', 'BCH', 'EOS', 'VERI', 'TRX',]].pct_change().corr())

    # scatter plot matrix
    sns.pairplot(closes.loc['2018', ['BTC', 'ETH', 'XRP', 'VERI', 'LTC']].pct_change().dropna())
