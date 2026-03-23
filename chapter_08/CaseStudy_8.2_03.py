import pandas as pd

df = pd.read_csv("data/audited_expressions.csv")

smp = df.sample(frac=1, random_state=42)
 
print("########### Audit these examples ########")
model1 = 0
model2 = 0
bothwrong = 0
samples = 2
for i in range(samples):
    print(f" GENERATOR: {smp.iloc[i]['CASE']}")
    print(f" AUDIT HAS IDIOM: {smp.iloc[i]['has_idiom']}")
    print(smp.iloc[i]['text'])
    print("----------------------------")
    valid1 = input("CASE RIGHT (y/n)")
    if valid1 == "y":
        model1 += 1
    valid2 = input("AUDIT RIGHT (y/n)")
    if valid2 == "y":
        model2 += 1
    if (valid1 == "n") & (valid2 == "n"):
        bothwrong += 1
    print("##############################################")

print("----------------------------------------")
print(f"Audit results")
print(f"Generating Model: {str(model1)} / {samples} Correct")
print(f"  Auditing Model: {str(model2)} / {samples} Correct")
print("----------------------------------------")
print(f" Both Models: {str(bothwrong)} / {samples} Errors")


