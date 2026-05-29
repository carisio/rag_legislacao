import json

# Caminhos dos arquivos
arquivo_entrada = './dados/dataset/corpus_qa.jsonl'
arquivo_saida = 'corpus_qa_embeddings_batch.jsonl'

with open(arquivo_entrada, 'r', encoding='utf-8') as f_in, \
     open(arquivo_saida, 'w', encoding='utf-8') as f_out:

    for linha in f_in:
        try:
            registro = json.loads(linha.strip())
            id_valor = registro.get('id')
            texto_valor = registro.get('questao_formatada')

            # Só processa se ambos os campos existirem
            if id_valor is not None and texto_valor is not None:
                novo_registro = {
                    "custom_id": id_valor,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": texto_valor
                    }
                }
                f_out.write(json.dumps(novo_registro, ensure_ascii=False) + '\n')
            else:
                print(f"Linha ignorada por falta de 'id' ou 'questao': {registro}")
        except json.JSONDecodeError:
            print("Erro ao decodificar linha. Ignorando.")

print(f"Arquivo '{arquivo_saida}' gerado com sucesso.")
