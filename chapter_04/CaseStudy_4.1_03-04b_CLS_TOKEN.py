from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch

BUCKET_PATH = "gs://ai_detection/"

df = pd.read_csv(BUCKET_PATH + "complete_with_features.csv")
texts = df['text'].tolist()

model_name = 'bert-base-uncased'
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def embed_text(text):
    # Tokenize the text
    inputs = tokenizer(text, truncation=True, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
       outputs = model(**inputs)
    h_states = outputs.last_hidden_state
    embedding = h_states[0, 0, :]
    return embedding.detach().cpu().numpy()


embeddings = [embed_text(x) for x in texts]

file_name = "Embeddings_CLS_" + model_name

np.save(file_name, embeddings)
