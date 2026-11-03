# Auditoria independente de percepcao, memoria e mundo

Campanha real unica: run 41551291622, tentativa 1. Fonte A `dc71247b2269d41c8d2aeb8c7d489977fac3ad3a`. Protocolo `93591ede5951438f92432fc6601773aaafe28b2ba5840309aa7bd3611170a324`.

ZIP verificado novamente: 50,240,431 bytes; SHA-256 `f396acfcbc8de35e67cbe79be4b6b986d7b536019bca2a62a3effa60ccefa129`. Os quatro CSV foram comparados byte a byte com as entradas desse ZIP e por SHA-256 com o manifesto autenticado.

**Resultado:** componentes numericos de percepcao e GRU passam; gate 7 completo fica `NOT_EVIDENCED` nesta auditoria por nao revisar o recibo de observacao. Gate 8 `FAIL`: previsao e beneficio do controle pelo mundo falham.

Nao foram executados Torch, ambiente, gradientes, novas seeds, inferencia de modelo ou rollouts. Foram usados CSV brutos, biblioteca padrao e NumPy, sem importar ou chamar o analisador do projeto.

## Cobertura e coerencia

1,735 episodios completos; 174,664 passos. Os 300 episodios shared-after exigem 39,688 transicoes: pares de acao e previsoes cobrem exatamente todas, sem duplicatas, faltas ou extras. O JSON enumera as 300 celulas stage/game/model/env e compara seu conjunto exato de passos.

Action pairs nao tem colunas stage/policy: sua origem shared-after foi associada por chave unica game/model/env/step, conferida com a identidade explicita das previsoes e o produtor congelado. Essa limitacao de rotulo permanece explicita.

246 terminos ambientais e 54 truncamentos. Nenhum truncamento foi tratado como terminal. Flags finais e somas de recompensa clipped conferem com os episodios. MAE e Brier foram recalculados a partir das previsoes escalares. Acoes, mascaras e mistura Q 0,8/0,2 conferem.

## Intervencoes de mesmo estado e ablaçoes

Percentuais sao a media das fracoes de mudanca das tres replicas. Pixels usa apenas as cinco seeds de ambiente predefinidas; GRU e mundo usam as vinte.

| Jogo | Pixels ocultos | GRU resetada | Mundo desligado |
|---|---:|---:|---:|
| snake | 83.907% | 37.222% | 9.408% |
| pong | 59.731% | 27.214% | 5.662% |
| agar | 50.075% | 48.259% | 7.591% |
| flappy | 57.036% | 22.085% | 3.451% |
| racing | 92.072% | 72.270% | 31.232% |

Os tres limiares de acao (pixels 5%, GRU 1%, mundo 1%) passam em 5/5 jogos; exigencia 4/5. O beneficio exige desempenho shared superior a ablaçao em pelo menos dois jogos: ganho medio >= 0, limite inferior do IC95 > 0 e ganho positivo em pelo menos duas replicas. Endpoints: comidas Snake/Agar, pontos liquidos Pong, canos Flappy e progresso normalizado Racing. Bootstrap pareado hierarquico: 10.000 repeticoes e seed 510002 com salt SHA-256 do rotulo congelado.

| Intervencao | Jogos com beneficio predefinido | Contagem | Resultado numerico |
|---|---|---:|---|
| pixels_hidden | snake, flappy | 2/5 | PASS |
| gru_reset | snake, flappy | 2/5 | PASS |
| world_off | nenhum | 0/5 | FAIL |

| Intervencao | Jogo | Ganho shared menos ablaçao | IC95 | Replicas favoraveis |
|---|---|---:|---|---:|
| pixels_hidden | snake | 3.400000 | [1.466667, 5.800000] | 3/3 |
| pixels_hidden | pong | 0.400000 | [-0.133333, 1.266667] | 2/3 |
| pixels_hidden | agar | 0.133333 | [-0.733333, 1.133333] | 1/3 |
| pixels_hidden | flappy | 1.133333 | [0.133333, 2.400000] | 3/3 |
| pixels_hidden | racing | 0.186851 | [-0.255361, 0.544509] | 2/3 |
| gru_reset | snake | 2.016667 | [0.916667, 3.266667] | 3/3 |
| gru_reset | pong | 0.183333 | [-0.233333, 0.783333] | 1/3 |
| gru_reset | agar | 0.216667 | [-0.150000, 0.583333] | 3/3 |
| gru_reset | flappy | 0.483333 | [0.033333, 1.100000] | 3/3 |
| gru_reset | racing | 0.016714 | [-0.042455, 0.104827] | 1/3 |
| world_off | snake | 0.200000 | [-0.833333, 1.133333] | 2/3 |
| world_off | pong | -0.016667 | [-0.133333, 0.083333] | 1/3 |
| world_off | agar | 0.016667 | [-0.183333, 0.183333] | 2/3 |
| world_off | flappy | 0.050000 | [-0.216667, 0.233333] | 2/3 |
| world_off | racing | -0.001290 | [-0.041883, 0.045521] | 1/3 |

## Previsao do mundo

Condicao fixa: MSE_model <= 0,95 x MSE_copy em cada jogo e replica, sem pooling para esconder falhas.

| Jogo | Modelo 1103 | Modelo 2207 | Modelo 3301 | Todas replicas |
|---|---:|---:|---:|---|
| snake | 0.896789 | 0.812333 | 0.824468 | PASS |
| pong | 0.987492 | 0.925595 | 0.908807 | FAIL |
| agar | 0.984119 | 0.978911 | 1.008826 | FAIL |
| flappy | 0.881026 | 0.863745 | 0.908793 | PASS |
| racing | 0.933840 | 1.119242 | 1.006572 | FAIL |

Apenas 9/15 celulas e 2/5 jogos cumprem a condicao. As seis falhas estao em Pong/1103, Agar/1103/2207/3301 e Racing/2207/3301.

Reward MAE melhora sobre TRAIN em 7/15 celulas; Brier terminal melhora em 14/15. Esses resultados descritivos nao substituem a condicao de MSE ou o beneficio de controle. Agar/2207 tem Brier pior que o baseline.

| Jogo | Replica | MAE modelo / TRAIN | Brier modelo / TRAIN | Suporte avaliaçao |
|---|---:|---|---|---:|
| snake | 1103 | 0.08571375 / 0.05524582 | 0.00373550 / 0.00733425 | 3460 |
| snake | 2207 | 0.08948453 / 0.05782079 | 0.00399930 / 0.00755935 | 3471 |
| snake | 3301 | 0.10476761 / 0.05424342 | 0.00155976 / 0.00589349 | 5522 |
| pong | 1103 | 0.04505136 / 0.03796934 | 0.00565024 / 0.00589131 | 3375 |
| pong | 2207 | 0.02905575 / 0.03713477 | 0.00463181 / 0.00600586 | 3310 |
| pong | 3301 | 0.04149736 / 0.04563659 | 0.00818404 / 0.00859874 | 2307 |
| agar | 1103 | 0.05513549 / 0.05376401 | 0.02037612 / 0.02329202 | 839 |
| agar | 2207 | 0.07425100 / 0.05528880 | 0.02328703 / 0.02214147 | 885 |
| agar | 3301 | 0.05066062 / 0.05875072 | 0.02503385 / 0.02525855 | 772 |
| flappy | 1103 | 0.06445163 / 0.06907869 | 0.01917946 / 0.02693021 | 725 |
| flappy | 2207 | 0.05486909 / 0.07251901 | 0.01360980 / 0.02576392 | 761 |
| flappy | 3301 | 0.05325512 / 0.06404409 | 0.00832228 / 0.01318842 | 1565 |
| racing | 1103 | 0.03884624 / 0.05719122 | 0.01556586 / 0.01863604 | 1056 |
| racing | 2207 | 0.00870103 / 0.00258650 | 0.00062491 / 0.00062667 | 6474 |
| racing | 3301 | 0.00880609 / 0.00606440 | 0.00148219 / 0.00155916 | 5166 |

Todos os baselines sao constantes por jogo/replica, rotulados TRAIN e suportados por 60.000 passos cada. O suporte e a taxa terminal foram reconstruidos em 15/15 celulas a partir do CSV de treino, excluindo step_limit e episodios parciais dos terminos. Seeds de treino e avaliacao nao se sobrepoem.

Media de recompensa clipped TRAIN: reconstruida em 12/15 celulas pelos retornos de treino e formulas congeladas. Snake/Pong/Racing tem recompensas dentro de [-1,1]; em Flappy os crossings recompensados sao recuperados do retorno para descontar o excedente de clipping de 0,005. Em Agar, o CSV agrega retornos sem preservar a distribuicao temporal de multiplas coletas/presas; reconstruir exatamente a soma clipped TRAIN e `NOT_EVIDENCED` nas tres replicas. Nao foi substituida por estimativa de avaliacao.

Os CSV de previsao contem os erros MSE por transicao, mas nao os tres vetores hidden de 64 coordenadas. A agregacao independente desses erros e verificavel; recalcular cada MSE a partir dos vetores e `NOT_EVIDENCED` neste conjunto de CSV. Nenhuma inferencia de checkpoint foi feita para preencher essa falta.

## O que os resultados sustentam

A GRU e o ramo de mundo existem na fonte congelada; as intervencoes registradas demonstram influencia sobre a escolha de acoes. Percepcao e memoria demonstram beneficio segundo o endpoint fixo em Snake e Flappy. Influencia do ramo de mundo nao demonstrou beneficio de controle: todos os cinco ICs de shared menos world_off incluem zero. A melhoria preditiva exigida tambem falha em seis celulas.

Ha apenas tres agentes treinados independentemente. Os intervalos seguem a predefinicao e continuam incertos. A auditoria cobre cinco simuladores locais congelados; nao demonstra dominio universal, Pong geral ou jogos externos. Gate 7 completo depende de observacao auditada separadamente; gate 8 permanece FAIL independentemente de recibos adicionais.

## Hashes de entrada

| Entrada | SHA-256 |
|---|---|
| episodes | `034ad7329d3faa93452b022a2c38ccc801bd39ee2068adf0f911e83e0d0e1f08` |
| action_pairs | `a98841203bec14abff2633d93a8b6264f4213118e852d030bc1fc4bb01c1f1be` |
| model_predictions | `b45a32ba039da29d83275c06be47b1975410bb5f39b8bd72dcebf9175ca75945` |
| training_curve | `2bc186e834cfa880159597179df792a5f4a80cf2e24c64a3ac20e9563d68e115` |
| data_provenance | `d1a4253d050947e9a15d84bb3c22accd21b7de0827b4b09119eaaa64852971c0` |
| root_manifest | `2797b5280d367be9e067d8528f2b0c8973081342eb248aacca19054475b18840` |
| protocol_snapshot | `f95e0ae4c7f0f0974173b1510a021579958e3d872fb5f107ecc20df758162be4` |

Os hashes das seis fontes de definicao congeladas e todos os numeradores, denominadores, ganhos por replica, ICs e verificacoes por celula constam do JSON complementar.
