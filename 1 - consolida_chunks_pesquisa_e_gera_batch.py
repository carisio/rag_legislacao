import os
import pandas as pd
import json
import tiktoken  # pip install tiktoken

# Caminho da pasta principal
pasta_base = 'dados\legislacao_csv'

# Lista para armazenar os DataFrames
dataframes = []

# Percorre todos os diretórios e subdiretórios
for pasta_atual, subpastas, arquivos in os.walk(pasta_base):
    for arquivo in arquivos:
        if arquivo.endswith('.csv'):
            caminho_completo = os.path.join(pasta_atual, arquivo)
            print(f"Processando arquivo: {caminho_completo}")
            try:
                df = pd.read_csv(caminho_completo, sep=',', encoding='utf-8')
                df['arquivo_origem'] = os.path.relpath(caminho_completo, pasta_base)
                dataframes.append(df)
            except Exception as e:
                print(f"Erro ao ler {caminho_completo}: {e}")

if dataframes:
    df_final = pd.concat(dataframes, ignore_index=True)

    # Remove duplicidades de urn, mantendo apenas a última ocorrência
    qtd_antes = len(df_final)

    df_final = (
        df_final
        .drop_duplicates(subset='urn', keep='last')
        .reset_index(drop=True)
    )

    qtd_depois = len(df_final)
    removidos = qtd_antes - qtd_depois

    print("Todos os arquivos CSV foram carregados e concatenados com sucesso.")
    print(f"{removidos} registros duplicados removidos com base na coluna 'urn'.")
else:
    print("Nenhum arquivo CSV encontrado.")

# Inicializa o tokenizer para o modelo text-embedding-3-small
# O tokenizer usado é o cl100k_base
tokenizer = tiktoken.get_encoding("cl100k_base")

# Caminho dos arquivos de saída
arquivo_embeddings  = 'legislacao_embeddings_batch.jsonl'
arquivo_base_pesquisa = 'base_pesquisa.jsonl'

with open(arquivo_embeddings, 'w', encoding='utf-8') as f_embeddings, \
     open(arquivo_base_pesquisa, 'w', encoding='utf-8') as f_base:
    for _, row in df_final.iterrows():
        urn = row['urn']
        texto = row['Texto']

        # =========================
        # Arquivo para embeddings
        # =========================
        
        # Tokeniza e trunca para 8000 tokens
        tokens = tokenizer.encode(texto)
        tokens_truncados = tokens[:8000]
        texto_truncado = tokenizer.decode(tokens_truncados)

        linha_embedding = {
            "custom_id": urn,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": texto_truncado
            }
        }
        f_embeddings.write(json.dumps(linha_embedding, ensure_ascii=False) + '\n')

        # =========================
        # Base de pesquisa completa
        # =========================
        linha_base = {
            "urn": urn,
            "text": texto,
            "tipo": row['Tipo']
        }

        f_base.write(
            json.dumps(linha_base, ensure_ascii=False) + '\n'
        )


# Caminho do arquivo de saída
arquivo_embeddings  = 'legislacao_apenas_artigos_sem_historico_embeddings_batch.jsonl'
arquivo_base_pesquisa = 'base_pesquisa_apenas_artigos_sem_historico.jsonl'

# Filtra apenas os artigos completos e remove o histórico (tudo dentro de [])
df_final_apenas_art = df_final[df_final["Tipo"] == "ART"].copy()
df_final_apenas_art_sem_historico = df_final_apenas_art.copy()
df_final_apenas_art_sem_historico["Texto"] = df_final_apenas_art_sem_historico["Texto"].str.replace(r"^\[.*?\]\s*", "", regex=True)

with open(arquivo_embeddings, 'w', encoding='utf-8') as f_embeddings, \
     open(arquivo_base_pesquisa, 'w', encoding='utf-8') as f_base:
    for _, row in df_final_apenas_art_sem_historico.iterrows():
        urn = row['urn']
        texto = row['Texto']

        # =========================
        # Arquivo para embeddings
        # =========================
        
        # Tokeniza e trunca para 8000 tokens
        tokens = tokenizer.encode(texto)
        tokens_truncados = tokens[:8000]
        texto_truncado = tokenizer.decode(tokens_truncados)

        linha_embedding = {
            "custom_id": urn,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": texto_truncado
            }
        }
        f_embeddings.write(json.dumps(linha_embedding, ensure_ascii=False) + '\n')

        # =========================
        # Base de pesquisa completa
        # =========================
        linha_base = {
            "urn": urn,
            "text": texto,
            "tipo": row['Tipo']
        }

        f_base.write(
            json.dumps(linha_base, ensure_ascii=False) + '\n'
        )



print(
    f"Arquivos '{arquivo_embeddings}' e "
    f"'{arquivo_base_pesquisa}' gerados com sucesso."
)
