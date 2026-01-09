"""
Configuration file for batch currency updates.
List the currency folder names you want to update.
These should match the folder names in the Crypto/ directory.
"""

# List of currency folder names to update
# These should match the folder names in Crypto/ (e.g., "Bitcoin", "AAVE", "Chainlink")
CURRENCIES_TO_UPDATE = [
    "Etherium/exponential_v2",  # Note: folder is "Etherium" (with 'i'), not "Ethereum"
    "Chainlink",  # Note: folder is "Chainlink" (lowercase 'l'), not "ChainLink"
    "Render",
    "Injection",
    "Litecoin",
    "Ton",
    "Doge",
]

# Output folder for consolidated results
OUTPUT_FOLDER = "Crypto/Update"

# Output file name for consolidated results
CONSOLIDATED_OUTPUT_FILE = "currency_summary.csv"

