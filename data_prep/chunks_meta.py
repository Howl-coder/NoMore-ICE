import os
import pandas as pd
from pathlib import Path

# Configuración: directorio raíz de tus textos por carpeta
BASE_DIR = Path(r"D:\DefenseGPT\data\raw\eng")



# Almacenes temporales
metadata_list = []
chunks_list = []

# Función: divide texto en bloques de máximo N caracteres
def chunk_text(text, max_chars=500):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# Recorrido de carpetas
for folder in BASE_DIR.iterdir():
    if folder.is_dir():
        for file in folder.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_id = f"eng_{len(metadata_list)+1:03}"
            chunked = chunk_text(content)

            # Metadatos del archivo original
            metadata_list.append({
                'file_id': file_id,
                'file_name': file.name,
                'title': file.stem.replace('_', ' ').title(),
                'topic': '',              # puedes rellenarlo luego
                'source_url': 'https://www.immigrantdefenseproject.org/',         # o agregarlo manualmente
                'format': 'txt',
                'language': 'eng',
                'folder': folder.name
            })

            # Chunks asociados
            for i, chunk in enumerate(chunked):
                chunks_list.append({
                    'chunk_id': f"{file_id}_chunk_{i+1}",
                    'file_id': file_id,
                    'text': chunk.strip()
                })

# Guardar CSVs
df_meta = pd.DataFrame(metadata_list)
df_chunks = pd.DataFrame(chunks_list)

df_meta.to_csv('./metadata_eng.csv', index=False)
df_chunks.to_csv('./chunks_eng.csv', index=False)

print("✅ CSVs generados: metadata_es.csv y chunks_es.csv")