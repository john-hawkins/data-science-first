import json
import pandas as pd

# Load the Excel file
# We modified the original to add entries for the 3 manufacturing categories
file_path = 'data/2022_NAICS_Descriptions_modified.xlsx'  
df = pd.read_excel(file_path, dtype={'Code': str})

def clean_text(txt):
    txt = str(txt)
    if txt[-1:]=="T":
        txt =  txt[:-1]
    return txt

df['Title'] = df['Title'].apply(clean_text)

# Get all root codes (2-digit)
root = df[df['Code'].str.len() == 2]

root_dict = pd.Series(root['Title'].values, index=root['Code']).to_dict()

with open("data/2022_NAICS_Roots.json", "w") as f:
    json.dump(root_dict, f)
 
lvl3 = df[df['Code'].str.len() == 3]

# Now get all categories below
for _, root_row in root.iterrows():
    root_code = root_row['Code']
    temp = lvl3[lvl3['Code'].str[0:2] == root_code]
    print(f"For {root_code} we have {len(temp)}")
    temp_dict = pd.Series(temp['Title'].values, index=temp['Code']).to_dict()
    file_name = "data/2022_NAICS_"+root_code+".json"
    with open(file_name, "w") as f:
        json.dump(temp_dict, f)



