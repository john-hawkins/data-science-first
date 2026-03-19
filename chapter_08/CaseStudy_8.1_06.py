"""
Very simple auditing script allows you to iterate over the selected subset
of generated data and indicate which of them are valid for the category.
"""
import pandas as pd

audit = df[(df['dim1']<5)&(df['dim2']>15)].copy()
others = audit[audit['CATEGORY']=='other'].copy()
 
print("########### Audit these examples ########")
invalid = 0
for i in range(len(others)):
    print("############# Category [other] ###############")
    print(others.iloc[i]['email'])
    print("##############################################")
    valid = input("Valid Email? (y/n)")
    if valid == "n":
        invalid += 1

print("----------------------------------------")
print(f"Audit results: {str(invalid)} / {str(len(others))} Errors")

