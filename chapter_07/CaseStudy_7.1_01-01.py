
import pandas as pd
from collections import defaultdict

# Load the Excel file
file_path = 'data/2022_NAICS_Descriptions.xlsx'  
df = pd.read_excel(file_path, dtype={'Code': str})

# Clean up any rows with missing codes
df = df.dropna(subset=['Code'])

# Filter only valid codes (2-6 digits and numeric)
df = df[df['Code'].str.match(r'^\d{2,6}$')]

# Get all root codes (2-digit)
root_nodes = df[df['Code'].str.len() == 2]
roots = {}
for _, root_row in root_nodes.iterrows():
    root_code = root_row['Code']
    title = root_row['Title'][:-1]
    if len(title)>27:
       title = title[:24] + "..."
    roots[root_code] = title

roots["31"] = "Manufacturing: Foods, beverages and fabrics"
roots["32"] = "Manufacturing: Wood, chemicals, petroleum and plastics"
roots["33"] = "Manufacturing: Metals, Refining, Processing, Equipment, Vehicles, Electronics"
roots["44"] = "Retail Trade - Vehicles, Homewares, Foods, Beverages" 
roots["45"] = "Retail Trade - Pharmaceuticals, Apparel, Lifestyle" 
roots["48"] = "Transportation and Support Services"
roots["49"] = "Warehousing, Delivery and Logistics"

# Create a dictionary to store counts
hierarchy_counts = defaultdict(lambda: {3: 0, 4: 0, 5: 0, 6: 0})

# Iterate through all entries
for _, row in df.iterrows():
    code = row['Code']
    if len(code) == 2:
        continue  # Skip root nodes themselves
    root_code = code[:2]  # First 2 digits determine the root node
    hierarchy_counts[root_code][len(code)] += 1

results = pd.DataFrame(columns=['Code', 'Title', 'Lvl3', 'Lvl4', 'Lvl5', 'Lvl6'])

keys = list(roots.keys())
keys.sort()
for root_code in keys:
    title = roots[root_code]
    counts = hierarchy_counts.get(root_code, {3: 0, 4: 0, 5: 0, 6: 0})
    record = {"Code":root_code, "Title":title, 'Lvl3':counts[3], 'Lvl4':counts[4], 'Lvl5':counts[5], 'Lvl6':counts[6]}
    results = pd.concat([results,pd.DataFrame([record])], ignore_index=True)

print(results.to_markdown(index=False))

