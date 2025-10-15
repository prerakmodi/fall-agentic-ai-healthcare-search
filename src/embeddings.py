from sentence_transformers import SentenceTransformer

import pandas as pd

df = pd.read_csv('illness.csv')
sentences = df['Content'].tolist()
model = SentenceTransformer('sentence-transformers/LaBSE')

embeddings = model.encode(sentences)
#print(embeddings)

similarities = model.similarity(embeddings, embeddings)
print(similarities)

