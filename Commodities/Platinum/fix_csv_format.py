"""One-off script to fix Platinum_historical_data.csv formatting."""
import csv

INPUT = "Platinum_historical_data.csv"
OUTPUT = "Platinum_historical_data.csv"

def round_val(s):
    if not s or not s.strip():
        return s
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s

with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

# Header
header = rows[0]
date_i = 0
value_i = 1
price_i = 2
open_i = 3
high_i = 4
low_i = 5

out = [header]
for r in rows[1:]:
    if not r:
        continue  # skip blank lines
    # Pad row to header length
    while len(r) < len(header):
        r.append("")
    # Historical rows: price was only in Value; copy to Price
    if (value_i < len(r) and price_i < len(r) and
        (not r[price_i] or not str(r[price_i]).strip()) and
        r[value_i] and str(r[value_i]).strip()):
        r[price_i] = r[value_i]
    # Newer Yahoo rows: close is in Price but Value was missing from export
    if (value_i < len(r) and price_i < len(r) and
        (not r[value_i] or not str(r[value_i]).strip()) and
        r[price_i] and str(r[price_i]).strip()):
        r[value_i] = r[price_i]
    # Round numeric columns
    for i in (value_i, price_i, open_i, high_i, low_i):
        if i < len(r) and r[i]:
            r[i] = round_val(r[i])
    out.append(r)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(out)

print(f"Wrote {len(out)} rows (header + {len(out)-1} data) to {OUTPUT}")
