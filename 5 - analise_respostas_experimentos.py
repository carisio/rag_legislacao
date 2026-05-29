import json
import glob
import pandas as pd
import os

# =========================
# 1) CARREGA O CORPUS
# =========================
corpus_file = "./dados/dataset/corpus_qa.jsonl"

dados = []
with open(corpus_file, "r", encoding="utf-8") as f:
    for linha in f:
        if not linha.strip():
            continue
        obj = json.loads(linha)
        dados.append({
            "id": obj.get("id"),
            "num_alternativas": obj.get("num_alternativas"),
            "questao_formatada": obj.get("questao_formatada"),
            "alternativa_correta": str(obj.get("alternativa_correta")).strip().lower()
        })

df = pd.DataFrame(dados)

# =========================
# 2) PROCESSA OS BATCHES
# =========================
arquivos_batch = glob.glob("./dados/experimentos/outputs/corrigidos/*.jsonl")

dfs_experimentos = []
for arquivo in arquivos_batch:
    nome_experimento = os.path.splitext(os.path.basename(arquivo))[0]
    
    linhas = []
    
    modelo_nome = None
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue

            obj = json.loads(linha)
            
            custom_id = obj.get("custom_id")

            # navegação segura
            try:
                body = obj["response"]["body"]
                modelo_nome = body.get("model")
                content = body["choices"][0]["message"]["content"]
            except Exception:
                continue
            
            if content is None or content in [""]:
                content = 'VAZIO'
            elif content not in ["A", "B", "C", "D", "E", "Certo", "Errado", "NAO_TEM"]:
                content = 'FORMATO_INADEQUADO'
                print(f'FORMATO_INADEQUADO: {custom_id}   {arquivo}')

            if custom_id is None or content is None:
                continue

            linhas.append({
                "id": custom_id,
                "resposta": content.strip().lower()
            })

    if not linhas or modelo_nome is None:
        continue

    df_experimento = pd.DataFrame(linhas)

    # renomeia a coluna "resposta" para o "nome do experimento"
    
    df_experimento = df_experimento.rename(columns={"resposta": nome_experimento})

    dfs_experimentos.append({
        "experimento": nome_experimento,
        "modelo": modelo_nome,
        "df": df_experimento
    })

# =========================
# 3) MERGE COM CORPUS
# =========================
df_final = df.copy()

for item in dfs_experimentos:
    df_experimento = item["df"]
    df_final = df_final.merge(df_experimento, on="id", how="left")

# =========================
# 4) AVALIAÇÃO
# =========================
resultados = []

for item in dfs_experimentos:
    nome_experimento = item['experimento']
    df_experimento = item['df']

    # comparação
    acertos = (
        df_final[nome_experimento] == df_final["alternativa_correta"]
    ).sum()

    total = df_final[nome_experimento].notna().sum()
    percentual = (acertos / total) * 100 if total > 0 else 0

    total_vazio = (df_final[nome_experimento] == "vazio").sum()
    total_formato_inadequado = (df_final[nome_experimento] == "formato_inadequado").sum()
    total_nao_tem_resposta_valida = (df_final[nome_experimento] == "nao_tem").sum()
    
    resultados.append({
        "experimento": nome_experimento,
        "acertos": acertos,
        "total": total,
        "total_vazio": total_vazio,
        "total_formato_inadequado": total_formato_inadequado,
        "total_resposta_invalida": total_nao_tem_resposta_valida,
        "percentual": percentual
    })

# =========================
# 5) BASELINE ALEATÓRIO
# =========================
df_final["chance"] = 1 / df_final["num_alternativas"]
acerto_esperado = df_final["chance"].sum()
percentual_esperado = (acerto_esperado / len(df_final)) * 100

# =========================
# PRINT RESULTADOS
# =========================
print("\n===== RESULTADOS POR EXPERIMENTO =====")
for r in resultados:
    print(f"\nExperimento: {r['experimento']}")
    print(f"Acertos: {r['acertos']} / {r['total']}")
    print(f"Acurácia: {r['percentual']:.2f}%")
    print(f"Sem resposta: {r['total_vazio']}")
    print(f"Resposta em formato inadequado: {r['total_formato_inadequado']}")
    print(f"Sem alternativa correta: {r['total_resposta_invalida']}")

print("\n===== BASELINE ALEATÓRIO =====")
print(f"Acertos esperados: {acerto_esperado:.2f} / {len(df_final)}")
print(f"Acurácia esperada: {percentual_esperado:.2f}%")