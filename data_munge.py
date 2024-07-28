import pandas as pd
import numpy as np
import os
import yfinance as yf
from fredapi import Fred
import glob
import openpyxl

# Read the raw data files
base_path = './Raw Price Data'
out_path = './Processed Data'
google_trends_path = './Google Trends Data/'
# Read shares outstanding data for each company
shares_files = {
    'AMD': './Shares Outstanding Data/AMD Quarterly Shares Outstanding.xlsx',
    'NVDA': './Shares Outstanding Data/NVIDIA Quarterly Shares Outstanding.xlsx'
}

companies = ['AMD', 'NVDA']
years = range(2019, 2025)

# Create a list to store all data
all_data = []

# Read and combine raw data files
for company in companies:
    for year in years:
        file_path = os.path.join(base_path, f'{company}_{year}.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, parse_dates=['DateTime'])
            data['Company'] = company  # Add company name
            all_data.append(data)
        else:
            print(f'File not found: {file_path}')

# Combine all data into a single DataFrame
df = pd.concat(all_data)

# Create a complete time index covering the full time range
start_time = df['DateTime'].min()
end_time = df['DateTime'].max()
full_time_index = pd.date_range(start=start_time, end=end_time, freq='T')

# Function to fill missing minutes
def fill_missing_minutes(group):
    group = group.set_index('DateTime').reindex(full_time_index)
    group['Company'] = group['Company'].fillna(method='ffill')
    group['Open'] = group['Open'].fillna(method='ffill')
    group['High'] = group['High'].fillna(method='ffill')
    group['Low'] = group['Low'].fillna(method='ffill')
    group['Close'] = group['Close'].fillna(method='ffill')
    group['Volume'] = group['Volume'].fillna(0)
    return group.reset_index().rename(columns={'index': 'DateTime'})

# Group by company and fill missing minutes
df = df.groupby('Company').apply(fill_missing_minutes).reset_index(drop=True)

# My API key for federal reserve database
fred_api_key = '8934d190e7d9586da2cdaf427b8f7094'
fred = Fred(api_key=fred_api_key)

# List the indices and economic indicators to get
indices = {
    'S&P 500': '^GSPC',
    'NASDAQ-100': '^NDX',
    'PHLX Semiconductor': '^SOX'
}

economic_indicators = {
    'Inflation Rate': 'CPIAUCSL',                     # Consumer Price Index (CPI) for all urban consumers
    'Federal Reserve Interest Rate': 'FEDFUNDS',      # Effective federal funds rate
    'Consumer Confidence Index': 'UMCSENT',           # University of Michigan: Consumer Sentiment
    'Effective Federal Fund Rate': 'FEDFUNDS'         # Effective federal funds rate (daily)
}

commodities = {
    'Oil Prices': 'CL=F',  # Crude Oil Futures
    'Gold Prices': 'GC=F'  # Gold Futures
}

# Get index data from yfinance
for name, ticker in indices.items():
    index_data = yf.download(ticker, start=start_time, end=end_time, interval='1d')['Close']
    index_data = index_data.resample('T').ffill().reindex(pd.date_range(start=start_time, end=end_time, freq='T')).ffill()
    df[name] = df['DateTime'].map(index_data)

# Get economic data from FRED
for name, series_id in economic_indicators.items():
    try:
        economic_data = fred.get_series(series_id, start_time, end_time)
        economic_data = economic_data.resample('T').ffill().reindex(pd.date_range(start=start_time, end=end_time, freq='T')).ffill()
        df[name] = df['DateTime'].map(economic_data)
    except ValueError as e:
        print(f"Error fetching data for {name} with series ID {series_id}: {e}")

# Get commodity prices from yfinance
for name, ticker in commodities.items():
    commodity_data = yf.download(ticker, start=start_time, end=end_time, interval='1d')['Close']
    commodity_data = commodity_data.resample('T').ffill().reindex(pd.date_range(start=start_time, end=end_time, freq='T')).ffill()
    df[name] = df['DateTime'].map(commodity_data)

# Process Google Trends data
def process_google_trends(file_path, company_name):
    gt_df = pd.read_csv(file_path, skiprows=2, parse_dates=['Day'])
    gt_df = gt_df.rename(columns={gt_df.columns[1]: 'Google Trends'})
    gt_df['Company'] = company_name
    return gt_df

# Get all Google Trends files
google_trends_files = glob.glob(os.path.join(google_trends_path, '*.csv'))

# Process each Google Trends file and concatenate them
gt_data = pd.concat([process_google_trends(file, 'AMD' if 'AMD' in file else 'NVDA') for file in google_trends_files])
gt_data = gt_data.drop_duplicates(subset=['Day', 'Company'])

# Merge Google Trends data with the main DataFrame
df['Date'] = pd.to_datetime(df['DateTime'].dt.date)
df = df.merge(gt_data, left_on=['Date', 'Company'], right_on=['Day', 'Company'], how='left')

# Forward fill the Google Trends data to match each minute
df['Google Trends'] = df['Google Trends'].fillna(method='ffill')

# Drop the 'Day' and 'Date' columns
df = df.drop(columns=['Day', 'Date'])

# Process shares outstanding data
def read_shares_data(file_path, company_name):
    shares_df = pd.read_excel(file_path, skiprows=2)
    shares_df.columns = ['Date', 'Number of Shares']
    shares_df['Date'] = pd.to_datetime(shares_df['Date'])
    shares_df['Number of Shares'] *= 1e6  # Convert from millions to actual number of shares
    shares_df['Company'] = company_name
    return shares_df

shares_data_list = [read_shares_data(file_path, company) for company, file_path in shares_files.items()]

# Concatenate all shares data into one DataFrame
shares_df = pd.concat(shares_data_list)
shares_df = shares_df.sort_values(by=['Company', 'Date'])

# Ensure both 'Date' columns are in the same format
df['Date'] = df['DateTime'].dt.date
df['Date'] = pd.to_datetime(df['Date'])

# Fill 'Number of Shares' for each company
df['Number of Shares'] = float('nan')

for company in shares_df['Company'].unique():
    company_shares = shares_df[shares_df['Company'] == company].reset_index(drop=True)
    for i in range(len(company_shares) - 1):
        start_date = company_shares.loc[i, 'Date']
        end_date = company_shares.loc[i + 1, 'Date']
        mask = (df['Date'] >= start_date) & (df['Date'] < end_date) & (df['Company'] == company)
        df.loc[mask, 'Number of Shares'] = company_shares.loc[i, 'Number of Shares']
    # Handle the last date range
    last_start_date = company_shares.loc[len(company_shares) - 1, 'Date']
    mask = (df['Date'] >= last_start_date) & (df['Company'] == company)
    df.loc[mask, 'Number of Shares'] = company_shares.loc[len(company_shares) - 1, 'Number of Shares']

# Create separate dataframes for AMD and NVDA
amd_df = df[df['Company'] == 'AMD'].copy()
nvda_df = df[df['Company'] == 'NVDA'].copy()

# Rename columns with company suffixes
amd_df = amd_df.rename(columns={
    'Open': 'AMD_Open', 'High': 'AMD_High', 'Low': 'AMD_Low', 'Close': 'AMD_Close', 'Volume': 'AMD_Volume',
    'Google Trends': 'AMD_Google Trends', 'Number of Shares': 'AMD_Number of Shares'
})

nvda_df = nvda_df.rename(columns={
    'Open': 'NVDA_Open', 'High': 'NVDA_High', 'Low': 'NVDA_Low', 'Close': 'NVDA_Close', 'Volume': 'NVDA_Volume',
    'Google Trends': 'NVDA_Google Trends', 'Number of Shares': 'NVDA_Number of Shares'
})

# Merge AMD and NVDA data on DateTime
merged_df = pd.merge(amd_df, nvda_df, on='DateTime', suffixes=('_AMD', '_NVDA'))

# Remove unnecessary columns and reorder
merged_df = merged_df.drop(columns=['Company_AMD', 'Date_AMD', 'Company_NVDA', 'Date_NVDA'])

# Define the columns to be divided by 4 for share split
columns_to_divide = ['NVDA_Open', 'NVDA_High', 'NVDA_Low', 'NVDA_Close']

# Set the datetime
cutoff_date = pd.to_datetime('2021-07-20 04:00:00')

# Apply the share split adjustment
merged_df.loc[merged_df['DateTime'] <= cutoff_date, columns_to_divide] /= 4

# Save the processed data file
output_file_path = os.path.join(out_path, 'final_data.csv')
merged_df.to_csv(output_file_path, index=False)

print(f'Processed data saved to {output_file_path}')
