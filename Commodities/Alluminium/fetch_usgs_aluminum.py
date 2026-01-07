"""
Fetch aluminum price data from USGS Mineral Industry Surveys.
USGS publishes monthly aluminum price data going back to the 1990s.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_usgs_data():
    """Try to fetch USGS aluminum price data."""
    print("=" * 60)
    print("Fetching Aluminum Price Data from USGS")
    print("=" * 60)
    
    # USGS Mineral Industry Survey archive
    base_url = "https://apps.usgs.gov/minerals-information-archives/aluminum/"
    
    print("\n[INFO] USGS publishes monthly Mineral Industry Surveys")
    print(f"       Archive: {base_url}")
    print("\n[INFO] These are typically PDF files that need manual extraction")
    print("       or specialized PDF parsing.")
    
    # Try to find recent reports
    try:
        response = requests.get(base_url, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for PDF links
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.I))
            print(f"\n  Found {len(pdf_links)} potential PDF reports")
            print("  [INFO] PDF extraction requires additional libraries (pdfplumber, PyPDF2)")
            print("  [INFO] Consider downloading PDFs manually and using a PDF parser")
    except Exception as e:
        print(f"  [WARN] Could not access USGS archive: {e}")
    
    print("\n" + "=" * 60)
    print("Recommendation:")
    print("=" * 60)
    print("1. Visit: https://apps.usgs.gov/minerals-information-archives/aluminum/")
    print("2. Download monthly PDF reports")
    print("3. Extract price data from PDFs (look for 'Average Price' tables)")
    print("4. Use add_missing_data.py to add the data")
    print("\nOR")
    print("Use a PDF parsing library to automate extraction from USGS PDFs")

if __name__ == "__main__":
    fetch_usgs_data()


