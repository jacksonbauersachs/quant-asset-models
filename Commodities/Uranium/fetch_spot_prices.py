"""
Programmatically fetch Uranium spot prices from various sources.
Tries multiple methods to get the missing months of data.
"""

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import time

def fetch_from_cameco():
    """Try to fetch spot prices from Cameco's investor page."""
    print("\n[1] Trying Cameco website...")
    try:
        url = "https://www.cameco.com/invest/markets/uranium-price"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for price data in various formats
        # Cameco typically shows spot prices in tables or specific divs
        prices = []
        
        # Try to find tables with price data
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    text = ' '.join(cell_texts)
                    
                    # Look for date pattern (but exclude if it's clearly a year like 2025)
                    date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text)
                    
                    # Look for price patterns - exclude years (1900-2100 range)
                    # Price should be reasonable (between $1 and $500 for uranium)
                    price_patterns = [
                        r'\$(\d{1,3}\.?\d{0,2})',  # $XX.XX format
                        r'(\d{1,2}\.\d{2})',  # XX.XX format (likely price, not year)
                    ]
                    
                    price = None
                    for pattern in price_patterns:
                        price_matches = re.finditer(pattern, text)
                        for match in price_matches:
                            candidate = float(match.group(1))
                            # Filter out years and unreasonable prices
                            if 1.0 <= candidate <= 500.0:  # Reasonable uranium price range
                                price = candidate
                                break
                        if price:
                            break
                    
                    if date_match and price:
                        date_str = date_match.group(1)
                        # Normalize date format
                        date_str = date_str.replace('/', '-')
                        prices.append({'Date': date_str, 'Price': price})
        
        # Also try to find price data in text/divs
        page_text = soup.get_text()
        # Look for patterns like "Spot Price: $XX.XX" or "U3O8: $XX.XX"
        price_patterns = [
            r'spot\s+price[:\s]+\$?(\d+\.?\d*)',
            r'U3O8[:\s]+\$?(\d+\.?\d*)',
            r'uranium[:\s]+\$?(\d+\.?\d*)',
        ]
        
        for pattern in price_patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE)
            for match in matches:
                price = float(match.group(1))
                # Try to find associated date nearby
                # This is a simplified approach
                pass
        
        if prices:
            print(f"  [OK] Found {len(prices)} price entries")
            return prices
        else:
            print("  [FAIL] Could not extract price data from Cameco page")
            return None
            
    except Exception as e:
        print(f"  [FAIL] Error fetching from Cameco: {e}")
        return None

def fetch_from_trading_economics():
    """Try to fetch from Trading Economics."""
    print("\n[2] Trying Trading Economics...")
    try:
        # Trading Economics uses JavaScript, so direct scraping is limited
        # But we can try their API endpoint if available
        url = "https://tradingeconomics.com/commodity/uranium"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Trading Economics loads data via JavaScript, so we'd need Selenium
        # For now, return None and suggest manual entry
        print("  [WARN] Trading Economics requires JavaScript (would need Selenium)")
        return None
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return None

def fetch_from_ycharts():
    """Try to fetch from YCharts."""
    print("\n[3] Trying YCharts...")
    try:
        url = "https://ycharts.com/indicators/uranium_spot_price"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # YCharts also uses JavaScript heavily
        print("  [WARN] YCharts requires JavaScript (would need Selenium)")
        return None
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return None

def try_yfinance_alternative():
    """Try using yfinance with uranium-related tickers that might track spot."""
    print("\n[4] Trying yfinance with uranium-related instruments...")
    try:
        import yfinance as yf
        
        # Some tickers that might correlate with spot prices
        # Note: These are ETFs/futures, not direct spot, but might be close
        tickers = ["URA", "URNM", "SRUUF"]
        
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start="2025-06-02", end=None)
                
                if not df.empty:
                    print(f"  [WARN] Found {ticker} data (ETF, not spot price)")
                    print(f"     Note: This is ETF price, not U3O8 spot price")
                    # Don't return this as it's not spot price
            except:
                continue
        
        return None
        
    except ImportError:
        print("  [FAIL] yfinance not available")
        return None
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return None

def fetch_spot_prices():
    """Main function to try all methods."""
    print("=" * 60)
    print("Fetching Uranium Spot Prices Programmatically")
    print("=" * 60)
    print("\nAttempting multiple sources...")
    
    all_prices = []
    
    # Try each method
    results = [
        fetch_from_cameco(),
        fetch_from_trading_economics(),
        fetch_from_ycharts(),
        try_yfinance_alternative(),
    ]
    
    # Collect any successful results
    for result in results:
        if result:
            all_prices.extend(result)
    
    if not all_prices:
        print("\n" + "=" * 60)
        print("[FAIL] Could not fetch spot prices programmatically")
        print("=" * 60)
        print("\nReason: Uranium spot prices are not easily accessible via free APIs.")
        print("Most sources require:")
        print("  - JavaScript rendering (need Selenium/Playwright)")
        print("  - Paid subscriptions (UxC)")
        print("  - Manual extraction from PDFs/reports")
        print("\nRecommendation:")
        print("  1. Visit Cameco's page manually: https://www.cameco.com/invest/markets/uranium-price")
        print("  2. Check their monthly investor reports for spot price tables")
        print("  3. Use the add_missing_data.py script to enter the prices")
        return None
    
    # Process and return prices
    if all_prices:
        df = pd.DataFrame(all_prices)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df.sort_values('Date')
        
        # Filter to only dates we need (after 2025-06-01)
        cutoff = pd.Timestamp('2025-06-01')
        df = df[df['Date'] > cutoff]
        
        if not df.empty:
            print(f"\n[OK] Found {len(df)} spot price entries")
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df
        else:
            print("\n[FAIL] No prices found for the needed date range")
            return None
    
    return None

def merge_with_existing(new_prices_df):
    """Merge fetched prices with existing CSV."""
    if new_prices_df is None or new_prices_df.empty:
        return None
    
    try:
        # Read existing data
        existing_df = pd.read_csv("Uranium_historical_data.csv")
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # Convert new prices dates
        new_prices_df['Date'] = pd.to_datetime(new_prices_df['Date'])
        
        # Combine
        combined_df = pd.concat([existing_df, new_prices_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Save
        combined_df.to_csv("Uranium_historical_data.csv", index=False)
        
        print(f"\n[OK] Successfully merged {len(new_prices_df)} new entries")
        print(f"  Total records: {len(combined_df)}")
        print(f"  Date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
        
        return combined_df
        
    except Exception as e:
        print(f"\n[FAIL] Error merging data: {e}")
        return None

if __name__ == "__main__":
    prices_df = fetch_spot_prices()
    
    if prices_df is not None and not prices_df.empty:
        print("\n" + "=" * 60)
        print("Fetched Prices:")
        print("=" * 60)
        print(prices_df.to_string(index=False))
        
        # Auto-merge if prices look valid
        if prices_df['Price'].min() >= 1.0 and prices_df['Price'].max() <= 500.0:
            print("\nPrices look valid. Auto-merging...")
            merge_with_existing(prices_df)
        else:
            print("\nWARNING: Prices may be invalid. Please review before merging.")
            print("Use add_missing_data.py to manually enter correct prices.")
    else:
        print("\nPlease use add_missing_data.py to manually enter the spot prices.")

