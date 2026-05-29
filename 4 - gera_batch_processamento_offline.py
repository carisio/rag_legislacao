import json

from pesquisa_vetorial import carregar_embeddings_hdf5,\
                                criar_indice_faiss,\
                                pesquisar_normalizado,\
                                filtrar_embeddings_apenas_artigos

# Entradas configuráveis
modelos_para_gerar = ["sabia-4-2026-01-06",
                      "sabiazinho-4-2026-01-06",
                      "gpt-5.4-2026-03-05",
                      "gpt-5.4-mini-2026-03-17",
                      "gpt-5.4-nano-2026-03-17"]

num_chunks_rag_para_gerar = [10]

# Arquivo jsonl que terá o batch
nome_input_batch_file = "./dados/experimentos/inputs/input_batch_{modelo}_{experimento}.jsonl"

msg_sistema_sem_contexto = """
Seu papel é resolver questões sobre a legislação brasileira. Para isso, você deverá obedecer às seguintes regras:

1. O usuário irá te passar uma questão que você deverá responder.
2. Essa questão pode ser de dois tipos. Pode ser de múltipla escolha, situação em que você deverá avaliar o enunciado e decidir por uma das alternativas apresentadas, ou pode ser de <CERTO> ou <ERRADO>, situação em que você deverá avaliar se o enunciado está correto ou não. O que diferencia as questões é a presença ou ausência de alternativas. Se houver alternativas, é de múltipla escolha. Caso contrário, é de <CERTO> ou <ERRADO>
3. A sua resposta deve ser direta, sem nenhuma marcação (pontuação, aspas ou qualquer outro caractere além da resposta).
3.1. Se for uma questão de múltiplica escolha, indique apenas a letra que responde ao enunciado.
3.2. Se for uma questão de <CERTO> ou <ERRADO> responda com "Certo" (para certo) ou "Errado" para errado.
""".strip()

msg_sistema_com_contexto = """
Seu papel é resolver questões sobre a legislação brasileira. Para isso, você deverá obedecer às seguintes regras:

1. O usuário irá te passar uma questão que você deverá responder.
2. Essa questão pode ser de dois tipos. Pode ser de múltipla escolha, situação em que você deverá avaliar o enunciado e decidir por uma das alternativas apresentadas, ou pode ser de <CERTO> ou <ERRADO>, situação em que você deverá avaliar se o enunciado está correto ou não. O que diferencia as questões é a presença ou ausência de alternativas. Se houver alternativas, é de múltipla escolha. Caso contrário, é de <CERTO> ou <ERRADO>
3. A sua resposta deve ser direta, sem nenhuma marcação (pontuação, aspas ou qualquer outro caractere além da resposta).
3.1. Se for uma questão de múltiplica escolha, indique apenas a letra que responde ao enunciado.
3.2. Se for uma questão de <CERTO> ou <ERRADO> responda com "Certo" (para certo) ou "Errado" para errado.

Além da questão, serão informados textos da legislação brasileira que podem ou não ser úteis para responder a questão e que poderão ser utilizados como apoio.
""".strip()

def get_msg_usuario(questao_formatada, query_emb, index, emb, num_chunks_rag):
    if num_chunks_rag is None:
        return questao_formatada
    else:
        chunks_proximos = pesquisar_normalizado(query_emb, index, emb, num_chunks_rag)
        chunks_proximos_numerados = [
            f"{i+1}. {chunks_proximos[i]['texto']}"
            for i in range(0, len(chunks_proximos))
        ]
        chunks_proximos_str = "\n\n".join(chunks_proximos_numerados)

        msg_usuario = (
            f"{questao_formatada}"
            "\n\n"
            "########################################\n"
            "Os trechos abaixo podem ser utilizados "
            "para auxiliar a resolução da questão:"
            "\n\n"
            f"{chunks_proximos_str}"
        )
        return msg_usuario

def gerar_arquivos_experimentos(index=None, emb=None, tipo_indice='', num_chunks_rag=None):
    print('#################')
    print(f'Gerando arquivos de experimentos para índice. Tipo do índice: {tipo_indice}')
    # Gera os arquivos de experimentos
    msg_sistema = msg_sistema_sem_contexto if num_chunks_rag is None else msg_sistema_com_contexto

    for nome_modelo in modelos_para_gerar:
        print(f"Gerando arquivo de experimento para modelo {nome_modelo} com {num_chunks_rag} chunks próximos...")
        nome_experimento = f"sem_contexto_{tipo_indice}" if num_chunks_rag is None else f"rag_{num_chunks_rag}_{tipo_indice}"
        
        with open(nome_input_batch_file.format(modelo=nome_modelo, experimento=nome_experimento), "w", encoding="utf-8") as fout:
            queries = emb["queries"]
            for query_id, query_texto, query_emb in zip(queries["ids"], queries["textos"], queries["embeddings"]):    
                
                msg_usuario = get_msg_usuario(query_texto, query_emb, index, emb, num_chunks_rag)
                
                nome_prop_max_tokens = "max_completion_tokens" if nome_modelo.startswith('gpt') else "max_tokens"
                requisicao = {
                    "custom_id": query_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": nome_modelo,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": msg_sistema},
                            {"role": "user", "content": msg_usuario}
                        ],
                        nome_prop_max_tokens: 1000
                    }
                }
        
                fout.write(json.dumps(requisicao, ensure_ascii=False) + "\n")
        
        print("Arquivo gerado")
            


########################################################################
# ABRIR A BASE DE CHUNKS HIERÁRQUICOS E GERAR OS SEGUINTES EXPERIMENTOS
# 1. Responder a pergunta sem nenhum contexto
# 2. Responder a pergunta passando 5 chunks hierárquicos
# 3. Responder a pergunta passando 5 chunks de artigos com histórico
########################################################################
arquivos_h5 = [
    f"./dados/dataset/text-embedding-3-small/chunks_hierarquicos_text-embedding-3-small_part_{i}.h5"
    for i in range(1, 5)
]
emb_hierarquicos = carregar_embeddings_hdf5(arquivos_h5)
# Cria índice de chunks hierárquicos
index_hierarquico = criar_indice_faiss(emb_hierarquicos)

# 1. Gera arquivos de experimentos sem nenhum contexto
#    Precisa de passar o emb, pois ele consulta as queries direto de lá
#    Tanto faz passar emb_completo ou emb_apenas_artigos para isso
gerar_arquivos_experimentos(None, emb_hierarquicos)

# 2. Gera arquivos de experimentos passando chunks hierárquicos
for num_chunks_rag in num_chunks_rag_para_gerar:
    gerar_arquivos_experimentos(index_hierarquico, emb_hierarquicos,
                                'hierarquico', num_chunks_rag)

# 3. Gera arquivos de experimentos passando apenas os artigos com histórico
emb_apenas_artigos_com_historico = filtrar_embeddings_apenas_artigos(emb_hierarquicos)
index_apenas_artigos_com_historico = criar_indice_faiss(emb_apenas_artigos_com_historico)

for num_chunks_rag in num_chunks_rag_para_gerar:
    gerar_arquivos_experimentos(index_apenas_artigos_com_historico, emb_apenas_artigos_com_historico,
                                'apenas_artigos_com_historico', num_chunks_rag)

########################################################################
# ABRIR A BASE DE CHUNKS APENAS ARTIGOS SEM HISTÓRICO E GERA O EXPERIMENTO:
# 4. Responder a pergunta passando 5 chunks de artigos sem histórico
########################################################################
arquivos_h5 = ["./dados/dataset/text-embedding-3-small/chunks_apenas_art_sem_historico_text-embedding-3-small_part_1.h5"]
emb_apenas_artigos_sem_historico = carregar_embeddings_hdf5(arquivos_h5)
index_apenas_artigos_sem_historico = criar_indice_faiss(emb_apenas_artigos_sem_historico)

for num_chunks_rag in num_chunks_rag_para_gerar:
    gerar_arquivos_experimentos(index_apenas_artigos_sem_historico, emb_apenas_artigos_sem_historico,
                                'apenas_artigos_sem_historico', num_chunks_rag)