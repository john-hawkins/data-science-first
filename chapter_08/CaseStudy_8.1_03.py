import random
from tqdm import tqdm
import pandas as pd
 

def get_list_from_file(file):
    filename = f"data/{file}"
    with open(filename, "r") as f:
        my_list = [line.strip() for line in f.readlines()]
    return my_list


def random_list_item(input):
   temp = random.randint(0, len(input)-1)
   return input[temp]


samples = 3
companies = get_list_from_file("companies.txt")
genders = get_list_from_file("genders.txt")
educations = get_list_from_file("educations.txt")
emotions = get_list_from_file("emotions.txt")

complaints = ["account", "network", "hardware", "other"]

results = pd.DataFrame()

for c in complaints:
    filename = f"{c}.txt"
    topic_list = get_list_from_file(filename)
 
    for t in tqdm(topic_list, desc=f"Generating for {filename}"):
        issue = t
        detail = ""
        if "Variations:".casefold() in t.casefold():
           temp = t.split("Variations:".casefold())
           issue = temp[0]
           if len(temp)> 1:
              variants = temp[1].split(",")
              vi = random.randint(0,len(variants)-1)
              detail = variants[vi]
        for i in range(0, samples):
            company = random_list_item(companies)
            gender = random_list_item(genders)
            age = str(random.randint(18,80))
            edu = random_list_item(educations)
            tenure = str(random.randint(1,10))
            emotion = random_list_item(emotions)
            record =                 {
                "COMPANY": company,
                "GENDER": gender,
                "AGE": age,
                "EDU":  edu,
                "TENURE": tenure,
                "CATEGORY": c,
                "ISSUE": issue,
                "DETAIL": detail,
                "EMOTION": emotion
            }

            try:
                response = chain.invoke(record)
                if 'email' in response: 
                    record['email'] = response['email']
                else:
                    record['email'] = "INVALID"
            except Exception as e:
                record['email'] = "ERROR"
                print("ERROR IN MODEL INVOCATION")
                print(e)
            results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)
    
results.to_csv("data/generated_complaint_emails_account.csv", index=False)

