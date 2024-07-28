import numpy as np

class StockData:
    def __init__(self, df, n_stocks=2):
        self.df = df
        self.n_stocks = n_stocks
        self.shape = self.get_state(0).shape

    def get_state(self, time_index):
        current_state = self.df.iloc[time_index][[
            'AMD_Open', 'AMD_High', 'AMD_Low', 'AMD_Close', 'AMD_Volume', 'AMD_Google Trends', 'AMD_Number of Shares',
            'NVDA_Open', 'NVDA_High', 'NVDA_Low', 'NVDA_Close', 'NVDA_Volume', 'NVDA_Google Trends', 'NVDA_Number of Shares',
            'S&P 500_AMD', 'NASDAQ-100_AMD', 'PHLX Semiconductor_AMD', 'Inflation Rate_AMD',
            'Federal Reserve Interest Rate_AMD', 'Consumer Confidence Index_AMD', 'Effective Federal Fund Rate_AMD',
            'Oil Prices_AMD', 'Gold Prices_AMD'
        ]]
        current_state['AMD_Number of Shares'] *= 1e-8
        current_state['NVDA_Number of Shares'] *= 1e-8
        current_state['S&P 500_AMD'] *= 1e-2
        current_state['NASDAQ-100_AMD'] *= 1e-2
        current_state['PHLX Semiconductor_AMD'] *= 1e-2

        return current_state.values

    def get_sell_price(self, time_index):
        sell_price = np.array([
            self.df.iloc[time_index]['AMD_Open'],
            self.df.iloc[time_index]['NVDA_Open']
        ])
        return sell_price

    def get_buy_price(self, time_index):
        buy_price = np.array([
            self.df.iloc[time_index]['AMD_Close'],
            self.df.iloc[time_index]['NVDA_Close']
        ])
        return buy_price