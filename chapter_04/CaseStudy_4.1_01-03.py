import pandas as pd
import numpy as np

# BASELINE ML MODELS
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# SUPPORT MODULES 
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/complete_with_features.csv")

features = ['text_len', 'text_wc', 'text_sc', 'text_lc', 'text_avg_wl', 
            'text_max_wl', 'text_cwd', 'text_caps', 'text_punc', 
            'text_misspelling', 'text_grammar_err'] 

train = df[df["RANDOM"]<0.8]
test = df[df["RANDOM"]>=0.8]

X_train = train.loc[:,features]
y_train = train.loc[:,"generated"] 
X_test = test.loc[:,features]
y_test = test.loc[:,"generated"]


preprocessor = make_pipeline(
   StandardScaler()
)

nb = ComplementNB()
lr = LogisticRegression(random_state=0)
xt = ExtraTreesClassifier()

lr_model = Pipeline(steps=[
   ('preprocessor', preprocessor),
   ('lr', lr )
])

nb.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
xt.fit(X_train, y_train)

## Compile a results dataset

results = pd.DataFrame(columns=["Model", "AUC", "Precision", "Recall"])

# Metrics for the Naive Bayes Model 
y_pred = nb.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = nb.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Model":"NaiveBayes", "AUC": auc, "Precision":prec, "Recall":recall}
results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)

# Metrics for the Logistic Regression Model
y_pred = lr_model.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = lr_model.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Model":"Logistic Regression", "AUC": auc, "Precision":prec, "Recall":recall}
results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)


# Metrics for the Extra Trees Model
y_pred = xt.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = xt.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Model":"Extra Trees", "AUC": auc, "Precision":prec, "Recall":recall}
results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)

results = results.round(3)

# Display Results DataFrame as a Markdown Table
markdown_table = results.to_markdown(index=False)
print(markdown_table)

