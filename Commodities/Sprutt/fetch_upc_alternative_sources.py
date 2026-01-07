"""
Fetch UPC historical data from multiple legitimate sources.
Tries various methods to get pre-2021 UPC data.
"""

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "UPC_historical_data.csv")

def try_alpha_vantage():
    """Try Alpha Vantage API (free tier available)."""
    print("\n[1] Trying Alpha Vantage API...")
    # Note: Would need API key, but let's check if we can find UPC data
    # Alpha Vantage uses different ticker formats
    print("  [INFO] Alpha Vantage requires API key (free tier available)")
    print("  [INFO] Ticker format might be different for TSX stocks")
    return None

def try_stockanalysis_scrape():
    """Try scraping from StockAnalysis.com (legitimate financial data site)."""
    print("\n[2] Trying StockAnalysis.com...")
    try:
        # StockAnalysis URL for UPC
        url = "https://stockanalysis.com/stocks/upc/history/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # StockAnalysis typically uses JavaScript to load data
            # We'd need Selenium for this, but let's check if there's static data
            print("  [INFO] StockAnalysis.com uses JavaScript - would need Selenium")
            print("  [INFO] Visit manually: https://stockanalysis.com/stocks/upc/history/")
            print("  [INFO] Export data and save as UPC_historical_data.csv")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    return None

def try_tsx_direct():
    """Try TSX (Toronto Stock Exchange) website directly."""
    print("\n[3] Trying TSX website...")
    try:
        # TSX historical data URL
        url = "https://www.tsx.com/trading/market-data/historical-data"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        print("  [INFO] TSX website requires manual search")
        print("  [INFO] Visit: https://www.tsx.com/trading/market-data/historical-data")
        print("  [INFO] Search for 'U' or 'Uranium Participation Corporation'")
        print("  [INFO] Download historical data and save as UPC_historical_data.csv")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    return None

def try_yahoo_alternative_formats():
    """Try different Yahoo Finance ticker formats for UPC."""
    print("\n[4] Trying alternative Yahoo Finance ticker formats...")
    
    # Try various formats
    ticker_variants = [
        "U.TO",      # Standard TSX format
        "U-TO",      # Alternative format
        "UPC.TO",    # Full name
        "UPC-TO",    # Full name alternative
        "U:TSX",     # Exchange suffix format
        "UPC:TSX",   # Full name with exchange
    ]
    
    for ticker in ticker_variants:
        try:
            print(f"  Trying {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period='max', end="2021-07-16")
            
            if not df.empty:
                result = pd.DataFrame({
                    'Date': df.index.strftime('%Y-%m-%d'),
                    'Price': df['Close'].values
                })
                print(f"  [OK] Found {len(result)} records using {ticker}")
                return result
        except Exception as e:
            continue
    
    print("  [FAIL] No data found with any ticker format")
    return None

def try_polygon_io():
    """Try Polygon.io API (free tier available for historical data)."""
    print("\n[5] Trying Polygon.io API...")
    print("  [INFO] Polygon.io offers free API with registration")
    print("  [INFO] Ticker format: U:TSX for TSX stocks")
    print("  [INFO] Visit: https://polygon.io/ and sign up for free tier")
    print("  [INFO] API endpoint: https://api.polygon.io/v2/aggs/ticker/U:TSX/range/1/day/2005-01-01/2021-07-16")
    return None

def try_finviz_scrape():
    """Try Finviz (financial data site)."""
    print("\n[6] Trying Finviz...")
    try:
        # Finviz might have UPC data
        url = "https://finviz.com/quote.ashx?t=U.TO"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Finviz typically has current data, not full history
            print("  [INFO] Finviz has current data but limited historical")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    return None

def try_marketwatch():
    """Try MarketWatch historical data."""
    print("\n[7] Trying MarketWatch...")
    try:
        # MarketWatch URL for UPC
        url = "https://www.marketwatch.com/investing/stock/u"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            print("  [INFO] MarketWatch may have historical data")
            print("  [INFO] Visit: https://www.marketwatch.com/investing/stock/u")
            print("  [INFO] Look for 'Historical Data' section and download")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    return None

def try_quandl():
    """Try Quandl/Nasdaq Data Link (some free data available)."""
    print("\n[8] Trying Quandl/Nasdaq Data Link...")
    print("  [INFO] Quandl has some free historical stock data")
    print("  [INFO] May require API key for TSX data")
    print("  [INFO] Visit: https://data.nasdaq.com/")
    return None

def try_web_scraping_alternative():
    """Try scraping from alternative financial data sites."""
    print("\n[9] Trying alternative web sources...")
    
    # Try Investing.com
    try:
        url = "https://www.investing.com/equities/uranium-participation-corp-historical-data"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            print("  [INFO] Investing.com may have historical data")
            print("  [INFO] Visit: https://www.investing.com/equities/uranium-participation-corp-historical-data")
            print("  [INFO] Download historical data table")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    
    return None

def try_stooq():
    """Try Stooq (free historical stock data)."""
    print("\n[10] Trying Stooq...")
    try:
        # Stooq format for TSX: U.TO or U-TO
        # Stooq CSV download format: https://stooq.com/q/d/l/?s=u.to&i=d
        url = "https://stooq.com/q/d/l/?s=u.to&i=d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200 and len(response.content) > 100:
            # Stooq returns CSV directly
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                # Stooq format: Date,Open,High,Low,Close,Volume
                if 'Date' in df.columns and 'Close' in df.columns:
                    # Filter to pre-reorg
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df[df['Date'] < pd.Timestamp("2021-07-17")]
                    
                    if not df.empty:
                        result = pd.DataFrame({
                            'Date': df['Date'].dt.strftime('%Y-%m-%d'),
                            'Price': df['Close'].values
                        })
                        print(f"  [OK] Found {len(result)} records from Stooq!")
                        return result
            except Exception as e:
                print(f"  [WARN] Error parsing Stooq data: {e}")
        
        print("  [INFO] Stooq may not have UPC data or uses different format")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    
    return None

def main():
    """Try all methods to fetch UPC data."""
    print("=" * 60)
    print("UPC Historical Data Fetcher - Multiple Sources")
    print("=" * 60)
    print("\nTrying multiple legitimate sources to fetch UPC data...")
    
    # Try all methods
    methods = [
        try_yahoo_alternative_formats,
        try_stooq,
        try_stockanalysis_scrape,
        try_tsx_direct,
        try_marketwatch,
        try_web_scraping_alternative,
        try_polygon_io,
        try_alpha_vantage,
    ]
    
    for method in methods:
        result = method()
        if result is not None and not result.empty:
            # Save the data
            result.to_csv(OUTPUT_FILE, index=False)
            print(f"\n[SUCCESS] Saved {len(result)} UPC records to {OUTPUT_FILE}")
            print(f"  Date range: {result['Date'].iloc[0]} to {result['Date'].iloc[-1]}")
            print(f"  Price range: ${result['Price'].min():.2f} to ${result['Price'].max():.2f}")
            return result
    
    print("\n" + "=" * 60)
    print("Could not automatically fetch UPC data")
    print("=" * 60)
    print("\nRecommended manual sources:")
    print("1. Stooq: https://stooq.com/q/d/l/?s=u.to&i=d (try downloading CSV)")
    print("2. Investing.com: https://www.investing.com/equities/uranium-participation-corp-historical-data")
    print("3. StockAnalysis.com: https://stockanalysis.com/stocks/upc/history/")
    print("4. TSX: https://www.tsx.com/trading/market-data/historical-data")
    print("\nAfter downloading, save as: UPC_historical_data.csv")
    print("Format: Date,Price (YYYY-MM-DD format)")

if __name__ == "__main__":
    main()

