from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch

df = pd.read_csv("data/complete_with_features.csv")

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
   # Tokenize the text
   inputs = tokenizer(text, truncation=True, return_tensors='pt')
   with torch.no_grad():
      outputs = model(**inputs)
   h_states = outputs.last_hidden_state
   embedding = torch.mean(h_states[0, :, :], dim=0)
   return embedding

embeddings = df['text'].apply(lambda x: embed_text(x))

file_name = "Embeddings_" + model_name
np.save(file_name, embeddings)

