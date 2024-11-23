import pickle

# Ruta del archivo LabelEncoder
label_encoder_path = r"C:\Users\jplv0\PycharmProjects\final-exam-pcd2024-autumn\training_pipeline\models\label_encoder2.pkl"

# Cargar el LabelEncoder desde el archivo
with open(label_encoder_path, "rb") as file:
    label_encoder = pickle.load(file)

# Asegurarse de que el LabelEncoder esté ajustado (fitted)
if not hasattr(label_encoder, 'classes_'):
    raise ValueError("El LabelEncoder no está ajustado. Asegúrate de entrenar el LabelEncoder antes de guardarlo.")

# Crear el diccionario con el mapeo del índice al nombre de la clase
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# Imprimir el diccionario
print(label_mapping)
