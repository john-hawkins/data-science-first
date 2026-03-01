from transformers import AutoTokenizer, AutoModel
import gc
import pandas as pd
import numpy as np
import torch

BUCKET_PATH = "gs://ai_detection/"

df = pd.read_csv(BUCKET_PATH + "permuted_text_samples.csv")
texts = df['text'].tolist()

print("DATA LOADED - RECORDS:", str(len(df)))

model_name = 'bert-base-uncased'
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"MODEL LOADED TO DEVICE:{device}")

def embed_text(text):
    # Tokenize the text
    inputs = tokenizer(text, truncation=True, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
       outputs = model(**inputs)
    h_states = outputs.last_hidden_state
    embedding = torch.mean(h_states[0, :, :], dim=0)
    return embedding.detach().cpu().numpy()

embeddings = [embed_text(x) for x in texts]

file_name = "permutation_embeddings" 

np.save(file_name, embeddings)

print("FILE SAVED - EXITING")

