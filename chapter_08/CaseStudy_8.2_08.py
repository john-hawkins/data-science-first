from datasets import ClassLabel, Value
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import evaluate
import torch
import sys


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

BUCKET_PATH = "gs://idiom_translations/"

train = pd.read_csv(BUCKET_PATH + "idiom_classifier_train.csv")
test = pd.read_csv(BUCKET_PATH + "idiom_classifier_test.csv")

train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)

hf_model = "google-bert/bert-base-multilingual-uncased"
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(hf_model)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], examples["translation"],
        padding='max_length', truncation=True,
        max_length=max_length, return_tensors="pt"
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

roc_auc_score = evaluate.load("roc_auc")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    p = precision_metric.compute(predictions=predictions, references=labels)
    r = recall_metric.compute(predictions=predictions, references=labels)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    auc = roc_auc_score.compute(prediction_scores=probs[:, 1], references=labels)
    return auc | p | r 


epochs = 10
dropout = 0.1
warmup=100

model = AutoModelForSequenceClassification.from_pretrained(
           hf_model, 
           num_labels=2, 
           hidden_dropout_prob=dropout,
           ignore_mismatched_sizes=True
)
dirname = f"./checkpoints/BERT_{dropout}_{warmup}_{epochs}"
training_args = TrainingArguments(
      output_dir=dirname, 
      num_train_epochs=epochs,                # total number of training epochs
      warmup_steps=warmup,                       # number of warmup steps for learning rate scheduler
      weight_decay=0.01,                      # strength of weight decay
      logging_dir='./logs',                   # directory for storing logs
      logging_steps=100,
      eval_strategy='epoch'
)
trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train_dataset,
      eval_dataset=tokenized_test_dataset,
      compute_metrics=compute_metrics,
)
trainer.train()
results = trainer.evaluate()
print(results)

# Create a DataLoader
batch_size = 16
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=False)
model.to(device)
model.eval()

all_preds = []
all_labels = []
all_scores = []
with torch.no_grad(): # Disable gradient calculation during evaluation
      for batch in dataloader:
         input_ids = batch['input_ids'].to(device)
         attention_mask = batch['attention_mask'].to(device)
         token_type_ids = batch['token_type_ids'].to(device)
         labels = batch['labels'].to(device)
         outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
         logits = outputs.logits.cpu().numpy()
         predictions = np.argmax(logits, axis=-1)
         probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
         probabilities = probs[:,1]
         all_preds.extend(predictions)
         all_labels.extend(labels.cpu().numpy())
         all_scores.extend(probabilities)

# Calculate metrics
auc_score = roc_auc_score.compute(prediction_scores=all_scores, references=all_labels)['roc_auc']
p = precision_metric.compute(predictions=all_preds, references=all_labels)['precision']
r = recall_metric.compute(predictions=all_preds, references=all_labels)['recall']
accuracy = accuracy_score(all_labels, all_preds)
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)


posies = { "positives" : np.sum(all_labels)}
records = {"records" : len(all_labels)}
pospred = {"preds" : np.sum(all_preds) }
print(f"Test AUC: {round(auc_score, 3)}")
print(f"Precision: {round(p, 3)}")
print(f"Recall: {round(r, 3)}")
print(f"Test Data has {posies} positive records of {records} - {pospred} predicted positive")

with open("BERT_Fine_Tune_Results.txt", "a") as f:
      f.write(f"Model : {hf_model}\n")
      f.write(f"Epochs: {epochs}\n")
      f.write(f"Dropout: {dropout}\n")
      f.write(f"Warmup: {warmup}\n")
      f.write(f"AUC: {round(auc_score, 3)}\n")
      f.write(f"Accuracy: {round(accuracy,3)}\n")
      f.write(f"Precision: {round(p, 3)}\n")
      f.write(f"Recall: {round(r, 3)}\n")
      f.write(f"---------------\n")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("idiom_transalation_ROC.png")

plt.close()

display = PrecisionRecallDisplay.from_predictions(
    all_labels, all_scores, name="BERT", plot_chance_level=True
)
_ = display.ax_.set_title("Idiom Classifier Precision-Recall Curve")
plt.savefig("idiom_transalation_PRC.png")



