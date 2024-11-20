import pandas as pd
import json

# Cargar el archivo JSON como diccionario
file_path = 'Data/raw_data/tickets_classification_eng.json'
with open(file_path, "r") as file:
    data = json.load(file)

# Convertir JSON a DataFrame
df = pd.json_normalize(data)

# Seleccionar columnas de interés
columns_to_keep = [
    '_source.complaint_what_happened', 
    '_source.product', 
    '_source.sub_product'
]
df = df[columns_to_keep]

# Renombrar columnas
df.columns = ['complaint_what_happened', 'category', 'sub_product']

# Crear nueva columna `ticket_classification`
df['ticket_classification'] = df['category'] + " + " + df['sub_product']

# Eliminar columnas redundantes
df = df.drop(columns=['category', 'sub_product'])

# Limpieza de datos
df['complaint_what_happened'].replace('', pd.NA, inplace=True)
df.dropna(subset=['complaint_what_happened', 'ticket_classification'], inplace=True)

# Reiniciar índice
df.reset_index(drop=True, inplace=True)

# Guardar DataFrame transformado como CSV
output_path = 'Data/processed/cleaned_tickets.csv'
df.to_csv(output_path, index=False)
