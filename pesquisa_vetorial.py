import h5py
import numpy as np
import faiss
import re

# =========================================================
# Função para abrir os arquivos HDF5 particionados
# =========================================================
def carregar_embeddings_hdf5(arquivos_h5):
    estrutura = {
        "queries": {
            "ids": [],
            "textos": [],
            "embeddings": []
        },
        "base_pesquisa": {
            "ids": [],
            "textos": [],
            "embeddings": []
        }
    }

    for arquivo in arquivos_h5:
        print(f"Abrindo: {arquivo}")
        with h5py.File(arquivo, "r") as h5f:

            # =================================================
            # Queries
            # =================================================
            estrutura["queries"]["ids"].extend([
                x.decode("utf-8") if isinstance(x, bytes) else x
                   for x in h5f["queries"]["ids"][:]
            ])

            estrutura["queries"]["textos"].extend([
                x.decode("utf-8") if isinstance(x, bytes) else x
                for x in h5f["queries"]["textos"][:]
            ])

            estrutura["queries"]["embeddings"].append(h5f["queries"]["embeddings"][:])

            # =================================================
            # Base pesquisa
            # =================================================
            estrutura["base_pesquisa"]["ids"].extend([
                    x.decode("utf-8") if isinstance(x, bytes) else x
                    for x in h5f["base_pesquisa"]["ids"][:]
            ])

            estrutura["base_pesquisa"]["textos"].extend([
                    x.decode("utf-8") if isinstance(x, bytes) else x
                    for x in h5f["base_pesquisa"]["textos"][:]
            ])

            estrutura["base_pesquisa"]["embeddings"].append(h5f["base_pesquisa"]["embeddings"][:])

    # =====================================================
    # Concatena embeddings
    # =====================================================
    estrutura["queries"]["embeddings"] = np.vstack(estrutura["queries"]["embeddings"])

    estrutura["base_pesquisa"]["embeddings"] = np.vstack(estrutura["base_pesquisa"]["embeddings"])

    return estrutura

# =========================================================
# Filtra apenas artigos "puros"
# A regex usada é para identificar a id que termina
# em art{num} ou art{num}-{num}
# Poderia ser feito também checando o Tipo == ART, mas
# essa informação não foi codificada no arquivo h5
# =========================================================
def filtrar_embeddings_apenas_artigos(emb):
    regex_artigo = re.compile(
        r"!art\d+(?:-\d+)?$",
        flags=re.IGNORECASE
    )
    # =====================================================
    # Máscara booleana
    # =====================================================
    mask = [
        bool(regex_artigo.search(urn))
        for urn in emb["base_pesquisa"]["ids"]
    ]

    # =====================================================
    # Índices válidos
    # =====================================================
    indices_validos = [
        i for i, ok in enumerate(mask)
        if ok
    ]

    # =====================================================
    # Nova estrutura
    # =====================================================
    emb_filtrado = {
        "queries": emb["queries"],

        "base_pesquisa": {
            "ids": [
                emb["base_pesquisa"]["ids"][i]
                for i in indices_validos
            ],
            "textos": [
                emb["base_pesquisa"]["textos"][i]
                for i in indices_validos
            ],
            "embeddings": emb["base_pesquisa"]["embeddings"][
                indices_validos
            ]
        }
    }

    # =====================================================
    # Estatísticas
    # =====================================================
    print(
        f"Total original: "
        f"{len(emb['base_pesquisa']['ids'])}"
    )
    print(
        f"Total artigos: "
        f"{len(emb_filtrado['base_pesquisa']['ids'])}"
    )

    return emb_filtrado

# =========================================================
# Converte estrutura vetorizada para estrutura indexada
# =========================================================
def converter_para_dicionario(estrutura_hdf5):
    embeddings = {
        "queries": {},
        "base_pesquisa": {}
    }

    # =====================================================
    # Queries
    # =====================================================
    queries_ids = estrutura_hdf5["queries"]["ids"]
    queries_textos = estrutura_hdf5["queries"]["textos"]
    queries_embeddings = estrutura_hdf5["queries"]["embeddings"]

    for idx in range(len(queries_ids)):
        query_id = queries_ids[idx]
        embeddings["queries"][query_id] = {
            "texto": queries_textos[idx],
            "embeddings": queries_embeddings[idx]
        }

    # =====================================================
    # Base pesquisa
    # =====================================================
    base_ids = estrutura_hdf5["base_pesquisa"]["ids"]
    base_textos = estrutura_hdf5["base_pesquisa"]["textos"]
    base_embeddings = estrutura_hdf5["base_pesquisa"]["embeddings"]
    for idx in range(len(base_ids)):
        chunk_id = base_ids[idx]
        embeddings["base_pesquisa"][chunk_id] = {
            "texto": base_textos[idx],
            "embeddings": base_embeddings[idx]
        }

    return embeddings

# =========================================================
# Cria índice FAISS
# =========================================================
def criar_indice_faiss(emb):
    print("Criando índice FAISS...")

    # =====================================================
    # Embeddings base
    # =====================================================
    embeddings_base = emb["base_pesquisa"]["embeddings"]

    # FAISS trabalha melhor com float32
    embeddings_base = embeddings_base.astype(np.float32)

    # =====================================================
    # Normalização L2
    # =====================================================
    faiss.normalize_L2(embeddings_base)

    # =====================================================
    # Cria índice cosine similarity
    # =====================================================
    dim = embeddings_base.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_base)

    print("Índice criado.")
    print(f"Total vetores: {index.ntotal}")

    return index

# =========================================================
# Pesquisa no índice
# =========================================================
def pesquisar(query_embedding, index, emb, top_k=5):

    # =====================================================
    # Prepara query
    # =====================================================
    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    # =====================================================
    # Normaliza query
    # =====================================================
    faiss.normalize_L2(query_embedding)

    # =====================================================
    # Busca
    # =====================================================
    scores, indices = index.search(query_embedding, top_k)

    # =====================================================
    # Monta resultados
    # =====================================================
    resultados = []

    for score, idx in zip(scores[0], indices[0]):
        urn = emb["base_pesquisa"]["ids"][idx]
        texto = emb["base_pesquisa"]["textos"][idx]
        resultados.append({
            "urn": urn,
            "texto": texto,
            "score": float(score)
        })

    return resultados

# =========================================================
# Verifica se uma URN engloba outra
# =========================================================
def urn_a_engloba_urn_b(urn_mais_geral, urn_mais_especifica):
    """
    Retorna True se urn_mais_geral engloba urn_mais_especifica.

    Exemplo:
    urn:...!art29 engloba
    urn:...!art29_cpt_ali5
    """
    return urn_mais_especifica.startswith(urn_mais_geral)

# =========================================================
# Normaliza resultados
# =========================================================
def normalizar_resultados(resultados, k):
    """
    Recebe lista ordenada por score decrescente.

    Mantém apenas dispositivos mais gerais,
    removendo redundâncias.

    Parâmetros:
    - resultados: lista de dicts
    - k: quantidade desejada

    Retorna:
    - lista normalizada
    """
    resultados_normalizados = []

    for candidato in resultados:
        urn_candidata = candidato["urn"]
        adicionar = True
        indices_para_remover = []
        # =================================================
        # Verifica relação com itens já selecionados
        # =================================================
        for idx, existente in enumerate(resultados_normalizados):
            urn_existente = existente["urn"]
            # ---------------------------------------------
            # Caso 1:
            # Já existe algo mais geral
            # ---------------------------------------------
            if urn_a_engloba_urn_b(urn_existente, urn_candidata):
                # Ex:
                # existente = art29
                # candidato = art29_cpt
                adicionar = False
                break

            # ---------------------------------------------
            # Caso 2:
            # Novo candidato é mais geral
            # ---------------------------------------------
            if urn_a_engloba_urn_b(urn_candidata, urn_existente):
                # Ex:
                # candidato = art29
                # existente = art29_cpt
                indices_para_remover.append(idx)

        # =================================================
        # Remove específicos englobados
        # =================================================
        for idx in reversed(indices_para_remover):
            resultados_normalizados.pop(idx)

        # =================================================
        # Adiciona candidato
        # =================================================
        if adicionar:
            resultados_normalizados.append(candidato)

        # =================================================
        # Para quando atingir k
        # =================================================
        if len(resultados_normalizados) >= k:
            break

    return resultados_normalizados

# =========================================================
# Pesquisa no índice
# =========================================================
def pesquisar_normalizado(query_embedding, index, emb, top_k=5):
    resultados = pesquisar(query_embedding, index, emb, 200*top_k)
    resultados = normalizar_resultados(resultados, top_k)
    return resultados
