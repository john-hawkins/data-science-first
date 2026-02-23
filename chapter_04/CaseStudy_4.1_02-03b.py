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

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/complete_with_features.csv")

features = ['text']

train = df[df["RANDOM"]<0.8]
test = df[df["RANDOM"]>=0.8]

X_train = train.loc[:,features]
y_train = train.loc[:,"generated"] 
X_test = test.loc[:,features]
y_test = test.loc[:,"generated"]

nb = ComplementNB()
lr = LogisticRegression(random_state=0)
xt = ExtraTreesClassifier()

tfidf = make_pipeline(
   TfidfVectorizer(max_features=200, stop_words='english')
)

preprocessor = ColumnTransformer(
    transformers=[
         ("text", tfidf, 'text'),
    ]
)

feats = preprocessor.fit_transform(X_train)

nb.fit(feats, y_train)
lr.fit(feats, y_train)
xt.fit(feats, y_train)

nb_model = Pipeline(steps=[
   ('tfidf', preprocessor),
   ('nb', nb )
])
lr_model = Pipeline(steps=[
   ('tfidf', preprocessor),
   ('lr', lr )
])
xt_model = Pipeline(steps=[
   ('tfidf', preprocessor),
   ('xt', xt )
])


## Compile a results dataset

results = pd.DataFrame(columns=["Model", "AUC", "Precision", "Recall"])

# Metrics for the Naive Bayes Model 
y_pred = nb_model.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = nb_model.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Model":"NaiveBayes (TFIDF)", "AUC": auc, "Precision":prec, "Recall":recall}
results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)

# Metrics for the Logistic Regression Model
y_pred = lr_model.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = lr_model.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Model":"Logistic Regression (TFIDF)", "AUC": auc, "Precision":prec, "Recall":recall}
results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)


# Metrics for the Extra Trees Model
y_pred = xt_model.predict(X_test)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
temp2 = xt_model.predict_proba(X_test)
auc = roc_auc_score(y_test, temp2[:,1])
record = {"Model":"Extra Trees (TFIDF)", "AUC": auc, "Precision":prec, "Recall":recall}
results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)

results = results.round(3)

# Display Results DataFrame as a Markdown Table
markdown_table = results.to_markdown(index=False)
print(markdown_table)

