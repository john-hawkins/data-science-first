import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BUCKET_PATH = "gs://ai_detection/"
df = pd.read_csv(BUCKET_PATH + "complete_with_features.csv")
texts = df['text'].tolist()

models = {
   "TaylorAI":"TaylorAI/bge-micro-v2",
   "MiniLm":"sentence-transformers/all-MiniLM-L6-v2"
}

for k in models.keys():
   model_name = k
   model_path = models[model_name]
   model = SentenceTransformer(model_path, device='cuda')
   embeddings = model.encode(texts)
   file_name = "Embeddings_" + model_name
   np.save(file_name, embeddings)

