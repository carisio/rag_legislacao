# README

## Estrutura dos dados

```text
./
├── dados/
│   ├── dataset/
│   │   ├── text-embedding-3-small/
│   │   |   ├── text-embedding-3-small_part_1.h5
│   │   |   ├── text-embedding-3-small_part_2.h5
│   │   |   ├── text-embedding-3-small_part_3.h5
│   │   |   └── text-embedding-3-small_part_4.h5
│   │   ├── corpus_qa.jsonl
│   │   └── base_pesquisa.jsonl
│   ├── experimentos/
│   │   ├── inputs/
│   │   ├── outputs/
│   │   |   ├── raw
│   │   |   └── corrigidos
│   ├── legislacao_csv/
```

## Scripts

### "1 - consolida_chunks_pesquisa_e_gera_batch.py"

Acessa a pasta <code>./dados/legislacao_csv</code> e gera o arquivo <code>base_pesquisa.jsonl</code>. Além disso, gera o arquivo <code>legislacao_embeddings_batch.jsonl</code>, contendo os batches para geração de embeddings usando a API de batches da OpenAI (interface via browser).

### "2 - gera_batch_questoes.py"

Acessa o arquivo <code>./dados/dataset/corpus_qa.jsonl</code> e gera o arquivo <code>corpus_qa_embeddings_batch.jsonl</code>, contendo os batches para geração de embeddings usando a API de batches da OpenAI (interace via browser).

### "3 - converter_emb_batch_texto_para_h5.py"

Os scripts <code>1 - consolida_chunks_pesquisa_e_gera_batch.py</code> e <code>2 - gera_batch_questoes.py</code> gera dois arquivos jsonl para gerar embeddings usando o modelo <code>text-embedding-3-small</code>.

São arquivos temporários que podem ser excluídos e apagados depois.

Para a geração dos embeddings com esse modelo, os arquivos batches foram submetidos via interface do browser. Em seguida, foi feito o download dos resultados. Os embeddings da legislação é um arquivo texto de aproximadamente 1.3 GB, muito grande para manter no repositório. Por isso, os embeddings em texto foram convertidos para float e salvos em 4 arquivos hdf5.

A conversão foi feita com o script <code>3 - converter_emb_batch_texto_para_h5.py</code> e a estrutura pode ser carregada usando a função <code>carregar_embeddings_hdf5</code>, disponível no arquivo <code>pesquisa_vetorial.py</code>.

### "pesquisa_vetorial.py"

Funções auxiliares para carregar os embeddings em índices e pesquisá-los.

### "4 - gera_batch_processamento_offline.py"

Gera os arquivos dos experimentos para serem submetidos via API batch para os LLMs. São 3 grupos de experimentos:

1. Responder as questões sem nenhum contexto
2. Responder as questões usando apenas os dados do artigos completos gerados usando o método de geração de chunks hierárquicos: ou seja, a ementa e o histórico (nome do capítulo, seção etc) é concatenado junto com o texto do artigo.
3. Responder as questões usando chunks hierárquicos

Além disso, é necessário gerar um outro experimento que é usando apenas o texto do artigo completo, sem a concatenação da ementa e histórico anterior.

#### Correção manual das respostas

Os experimentos foram feitos pedindo a resposta direta. Nem todos os LLMs obedeceram isso. Então foi feita uma correção manual das respostas dos LLMs.

Por exemplo:
- se a resposta foi "A)", foi convertida para "A"
- se a resposta foi "Certo. Isso porque ...", foi convertido para "Certo"
- se a resposta foi "E) I, II e III", foi convertido para "E"
- se a resposta foi "Apenas os itens I e III" e isso se refere a uma alternativa válida, foi convertido para a letra da alternativa. Se não se referir a uma alternativa válida, foi alterado para "" (vazio)

Ou seja, foram feitos ajustes de forma, mantendo o conteúdo.

### "5 - analise_respostas_experimentos.py"

Verifica os acertos dos experimentos