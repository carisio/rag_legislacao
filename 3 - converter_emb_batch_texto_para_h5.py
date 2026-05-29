import json
import math
import h5py
import numpy as np
from tqdm import tqdm

# =========================================================
# Caminhos dos arquivos
# =========================================================
CHUNKS_HIERARQUICOS = {
    "ARQ_EMB_LEGIS": "embeddings_gerados_legislacao.jsonl",
    "ARQ_EMB_QA": "embeddings_gerados_corpus_qa.jsonl",
    "NOME_DATASET": "chunks_hierarquicos_text-embedding-3-small",
    "NUM_PARTES": 4,
    
    "ARQ_CORPUS_QA": "./dados/dataset/corpus_qa.jsonl",
    "ARQ_BASE_PESQUISA": "./dados/dataset/base_pesquisa.jsonl"
}

CHUNKS_APENAS_ARTIGOS_SEM_HISTORICO = {
    "ARQ_EMB_LEGIS": "embeddings_gerados_legislacao.jsonl",
    "ARQ_EMB_QA": "embeddings_gerados_corpus_qa.jsonl",
    "NOME_DATASET": "chunks_apenas_art_sem_historico_text-embedding-3-small",
    "NUM_PARTES": 1,
    
    "ARQ_CORPUS_QA": "./dados/dataset/corpus_qa.jsonl",
    "ARQ_BASE_PESQUISA": "./dados/dataset/base_pesquisa_apenas_artigos_sem_historico.jsonl"
}

#ARQ_EMB_LEGIS = "embeddings_gerados_legislacao.jsonl"
#ARQ_EMB_QA = "embeddings_gerados_corpus_qa.jsonl"

#ARQ_CORPUS_QA = "./dados/dataset/corpus_qa.jsonl"
#ARQ_BASE_PESQUISA = "./dados/dataset/base_pesquisa.jsonl"

# =========================================================
# Configurações
# =========================================================
DTYPE_EMBEDDING = np.float16

COMPRESSAO = "gzip"
NIVEL_COMPRESSAO = 9

#NUM_PARTES = 4

# =========================================================
# Função para carregar JSONL
# =========================================================
def carregar_jsonl(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()

            if not linha:
                continue

            yield json.loads(linha)

# =========================================================
# Função para dividir listas
# =========================================================
def dividir_em_partes(lista, num_partes):
    tamanho = math.ceil(len(lista) / num_partes)

    return [
        lista[i:i + tamanho]
        for i in range(0, len(lista), tamanho)
    ]

# =========================================================
# Carrega embeddings legislação
# =========================================================
def carregar_embeddings_de_arq_texto(ARQ_EMB):
    print("Carregando embeddings da legislação...")
    
    embs = {}
    for item in tqdm(carregar_jsonl(ARQ_EMB)):
        custom_id = item["custom_id"]
        embedding = np.array(
            item["response"]["body"]["data"][0]["embedding"],
            dtype=DTYPE_EMBEDDING
        )
        embs[custom_id] = embedding
    return embs

# =========================================================
# Carrega corpus QA
# =========================================================
def carregar_corpus_qa(emb_qa, ARQ_CORPUS_QA):
    print("Carregando corpus QA...")
    
    queries_ids = []
    queries_textos = []
    queries_embeddings = []
    
    for item in tqdm(carregar_jsonl(ARQ_CORPUS_QA)):
        query_id = item["id"]
    
        if query_id not in emb_qa:
            print(f"[WARNING] Embedding não encontrado: {query_id}")
            continue
    
        texto = item["questao_formatada"]
    
        queries_ids.append(query_id)
        queries_textos.append(texto)
        queries_embeddings.append(emb_qa[query_id])
    
    return queries_ids, queries_textos, queries_embeddings

# =========================================================
# Carrega base pesquisa
# =========================================================
def carregar_base_pesquisa(emb_legislacao, ARQ_BASE_PESQUISA):
    print("Carregando base pesquisa...")
    base_ids = []
    base_textos = []
    base_embeddings = []
    
    for item in tqdm(carregar_jsonl(ARQ_BASE_PESQUISA)):
        urn = item["urn"]
    
        if urn not in emb_legislacao:
            print(f"[WARNING] Embedding não encontrado: {urn}")
            continue
    
        texto = item["text"]
    
        base_ids.append(urn)
        base_textos.append(texto)
        base_embeddings.append(emb_legislacao[urn])
    
    return base_ids, base_textos, base_embeddings

def gerar_dataset_hdf5(config):
    emb_legislacao = carregar_embeddings_de_arq_texto(config['ARQ_EMB_LEGIS'])
    emb_qa = carregar_embeddings_de_arq_texto(config['ARQ_EMB_QA'])
    
    queries_ids, queries_textos, queries_embeddings = carregar_corpus_qa(emb_qa, config['ARQ_CORPUS_QA'])
    base_ids, base_textos, base_embeddings = carregar_base_pesquisa(emb_legislacao, config['ARQ_BASE_PESQUISA'])

    # =========================================================
    # Divide em partes
    # =========================================================
    NUM_PARTES = config['NUM_PARTES']
    queries_ids_parts = dividir_em_partes(queries_ids, NUM_PARTES)
    queries_textos_parts = dividir_em_partes(queries_textos, NUM_PARTES)
    queries_embeddings_parts = dividir_em_partes(queries_embeddings, NUM_PARTES)
    
    base_ids_parts = dividir_em_partes(base_ids, NUM_PARTES)
    base_textos_parts = dividir_em_partes(base_textos, NUM_PARTES)
    base_embeddings_parts = dividir_em_partes(base_embeddings, NUM_PARTES)
    
    # =========================================================
    # Salva cada parte
    # =========================================================
    for parte_idx in range(NUM_PARTES):
    
        nome_arquivo = f"{config['NOME_DATASET']}_part_{parte_idx + 1}.h5"
    
        print(f"\nSalvando {nome_arquivo}...")
        # =====================================================
        # Converte embeddings para matrizes NumPy
        # =====================================================
        queries_emb_array = np.vstack(queries_embeddings_parts[parte_idx])
    
        base_emb_array = np.vstack(base_embeddings_parts[parte_idx])
    
        # =====================================================
        # Salva HDF5
        # =====================================================
        with h5py.File(nome_arquivo, "w") as h5f:
            # =================================================
            # Queries
            # =================================================
            grp_queries = h5f.create_group("queries")
    
            grp_queries.create_dataset(
                "ids",
                data=np.array(
                    queries_ids_parts[parte_idx],
                    dtype=h5py.string_dtype(encoding="utf-8")
                ),
                compression=COMPRESSAO,
                compression_opts=NIVEL_COMPRESSAO
            )
    
            grp_queries.create_dataset(
                "textos",
                data=np.array(
                    queries_textos_parts[parte_idx],
                    dtype=h5py.string_dtype(encoding="utf-8")
                ),
                compression=COMPRESSAO,
                compression_opts=NIVEL_COMPRESSAO
            )
    
            grp_queries.create_dataset(
                "embeddings",
                data=queries_emb_array,
                compression=COMPRESSAO,
                compression_opts=NIVEL_COMPRESSAO,
                chunks=True
            )
    
            # =================================================
            # Base pesquisa
            # =================================================
            grp_base = h5f.create_group("base_pesquisa")
    
            grp_base.create_dataset(
                "ids",
                data=np.array(
                    base_ids_parts[parte_idx],
                    dtype=h5py.string_dtype(encoding="utf-8")
                ),
                compression=COMPRESSAO,
                compression_opts=NIVEL_COMPRESSAO
            )
    
            grp_base.create_dataset(
                "textos",
                data=np.array(
                    base_textos_parts[parte_idx],
                    dtype=h5py.string_dtype(encoding="utf-8")
                ),
                compression=COMPRESSAO,
                compression_opts=NIVEL_COMPRESSAO
            )
    
            grp_base.create_dataset(
                "embeddings",
                data=base_emb_array,
                compression=COMPRESSAO,
                compression_opts=NIVEL_COMPRESSAO,
                chunks=True
            )
    
        print(f"{nome_arquivo} salvo com sucesso.")
    
        print(f"Queries: {len(queries_ids_parts[parte_idx])}")
    
        print(f"Base pesquisa: {len(base_ids_parts[parte_idx])}")
    
    # =========================================================
    # Final
    # =========================================================
    print("\nProcesso concluído.")
    
gerar_dataset_hdf5(CHUNKS_APENAS_ARTIGOS_SEM_HISTORICO)