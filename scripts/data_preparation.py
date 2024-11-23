# data_preparation.py

import pandas as pd
import numpy as np
import pickle
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder

def load_preprocessed_data():
    print("Cargando el dataset preprocesado...")
    df = pd.read_csv('data/processed_data/cld_dataset.csv')
    print(f"Dataset cargado con {len(df)} registros.")
    return df

def generate_embeddings(df):
    print("Generando embeddings con BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []
    total_texts = len(df['cleaned_text'])
    for idx, text in enumerate(df['cleaned_text']):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding)
        
        # Mostrar progreso cada 100 registros
        if (idx + 1) % 100 == 0:
            print(f"Embeddings generados para {idx + 1}/{total_texts} textos...")
    
    embeddings = np.vstack(embeddings)
    np.save('data/processed_data/embeddings.npy', embeddings)
    print("Embeddings guardados en 'data/processed_data/embeddings.npy'.")
    return embeddings

def encode_labels(df):
    print("Codificando etiquetas...")
    le = LabelEncoder()
    y = le.fit_transform(df['ticket_classification'])
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("LabelEncoder guardado en 'models/label_encoder.pkl'.")
    return y

if __name__ == '__main__':
    df = load_preprocessed_data()
    embeddings = generate_embeddings(df)
    y = encode_labels(df)
    print("Proceso completado exitosamente.")
