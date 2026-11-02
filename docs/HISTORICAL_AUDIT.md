# Auditoria histórica — Fases científicas 1–4

Consulta ao estado real do GitHub em 2026-11-01, America/Sao_Paulo (2026-11-02 UTC). Esta auditoria leu as quatro PRs, todos os comentários de conversa retornados pelo conector, os relatórios científicos disponíveis, metadados dos quatro runs oficiais e dos seis artefatos científicos. Não executou treinamento nem episódios científicos. A conferência dos metadados abaixo não substitui a verificação dos bytes preservados e dos manifestos internos.

## Estado e cadeia de commits

As PRs [#1](https://github.com/Myrroth/NOVA-Neural-Arena/pull/1), [#2](https://github.com/Myrroth/NOVA-Neural-Arena/pull/2), [#3](https://github.com/Myrroth/NOVA-Neural-Arena/pull/3) e [#4](https://github.com/Myrroth/NOVA-Neural-Arena/pull/4) estão abertas, em rascunho, sem merge. Elas formam uma cadeia: `main` → laboratório Snake → Fase 2 → Fase 3 → Fase 4. A branch desta entrega parte da Fase 4; isso consolida ferramentas para revisão sem integrar as PRs anteriores em `main`.

| Fase | Commit executável congelado | HEAD da infraestrutura oficial |
|---|---|---|
| 1: piloto Snake/DQN | `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463` | `f49adc6fc45596c81169691aecb1e0937e201ea2` |
| 2: controles neurais não treinados | `f9e1b171a8c045f8bd7c874432114e4bc654665e` | `891d26b60e82fb8ae49eac17465064d58f3f0554` |
| 3: tamanhos 8/10/12 | `8892980eccfb623de30f3803a7550aa1fa31a9c6` | `b37322add54d2751e16fc7db9177d1c7ef41817d` |
| 4: 12 estados iniciais | `8b4b6696e980191349e803218af6e6d55d8ff7e9` | `af4bd307c6b6a75980a1b66c007cfd6a1d9a5d59` |

Comparações locais entre esses commits confirmaram ausência de diferenças no Snake, DQN, treinador e benchmark originais entre o piloto e o HEAD da Fase 4. Cada avaliador das Fases 2–4 também permanece idêntico ao respectivo commit congelado. Mudanças futuras devem entrar em módulos novos; os commits executáveis históricos não serão reescritos.

## Fontes científicas preservadas em texto

O snapshot [history/prs-1-4-snapshot.json](history/prs-1-4-snapshot.json) contém os corpos e metadados das PRs e todos os comentários retornados, com IDs e URLs de origem. Ele preserva também as decisões intermediárias, que descrevem estados anteriores à execução.

| Documento encontrado | Link original | Interpretação correta |
|---|---|---|
| Parecer preliminar do Chat 4, sem número no título | [PR #1, 6563751208](https://github.com/Myrroth/NOVA-Neural-Arena/pull/1#issuecomment-6563751208) | Evidência então insuficiente; CI não demonstra eficácia. Possível documento referido como nº 001, mas a numeração não foi confirmada. |
| Relatório científico 002 | [PR #1, 6564156873](https://github.com/Myrroth/NOVA-Neural-Arena/pull/1#issuecomment-6564156873) | Piloto contra aleatória em Snake 10×10; sem controle neural inicial e sem comparação comum de seeds entre os três treinos. |
| Parecer científico da Fase 2, sem número no título | [PR #2, 6566355671](https://github.com/Myrroth/NOVA-Neural-Arena/pull/2#issuecomment-6566355671) | Evidência de aprendizado contra redes não treinadas, na mesma distribuição. Possível nº 003, sem confirmação literal. |
| Relatório Científico nº 004 | [PR #3, 6566798413](https://github.com/Myrroth/NOVA-Neural-Arena/pull/3#issuecomment-6566798413) | Generalização zero-shot limitada à mudança de dimensões no mesmo Snake. |
| Relatório Científico nº 005 | [PR #4, 6567175661](https://github.com/Myrroth/NOVA-Neural-Arena/pull/4#issuecomment-6567175661) | Robustez limitada aos 12 estados centrais testados; ausência de diferença detectada não prova equivalência. |

Não renumerar os documentos silenciosamente. Gráficos e CSVs derivados do relatório 002 são mencionados como gerados separadamente no Chat 4, sem estarem anexados ao repositório; sua localização persistente não foi confirmada nesta consulta.

## Inventário oficial

As quatro APIs de runs retornaram `status=completed`, `conclusion=success`, `run_attempt=1`. As seis entradas de artefato retornaram `expired=false`; digests da API coincidem com os publicados nos relatórios operacionais e científicos. O snapshot [history/actions-runs-summary.json](history/actions-runs-summary.json) preserva identificação/estado dos runs; [history/actions-artifacts-metadata.json](history/actions-artifacts-metadata.json) preserva metadados de artefatos, tamanhos e expirações UTC exatas.

| Fase / seed | Run | Artefato | Bytes ZIP | Expira em America/Sao_Paulo |
|---|---|---|---:|---|
| 1 / 11 | [41350959549](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549) | [13101471957](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549/artifacts/13101471957) | 79.401 | 01/12/2026 18:15:00 |
| 1 / 42 | mesmo run | [13101072100](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549/artifacts/13101072100) | 77.408 | 01/12/2026 18:15:02 |
| 1 / 73 | mesmo run | [13100988601](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549/artifacts/13100988601) | 80.840 | 01/12/2026 18:14:59 |
| 2 | [41367188257](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41367188257) | [13106764292](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41367188257/artifacts/13106764292) | 5.864 | 01/12/2026 21:27:36 |
| 3 | [41370622154](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370622154) | [13108893868](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370622154/artifacts/13108893868) | 15.513 | 01/12/2026 22:06:38 |
| 4 | [41373651632](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41373651632) | [13110740130](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41373651632/artifacts/13110740130) | 51.492 | 01/12/2026 22:40:58 |

SHA-256 dos ZIPs originais, sem recompressão:

```text
13101471957 f8e3ea85e2e7a01a2522ab24d7e96fa024d0232dec71f0f3171af25441ec1a74
13101072100 3bc28cc078a38323539090284c80976f53351cd894a10882a2d7138e52359642
13100988601 0c2dcde1586a3be8e3c55b3ff67c67d4f56dca53012b96b0d690db517ad54b48
13106764292 cb74f81d59c3eeb37111998fa977346af381e3013c3116c6065e4e1569eed4cf
13108893868 75c12daa2f5b0a126eed3d0ca8b72c9938d22f937e6911f27a5b5b5dcc83c2c4
13110740130 dc4bb9640b5e54ce3dcc53ea7b766c880097810ca37ec4e0532549dd75dc9289
```

SHA-256 dos checkpoints `model.pt` originais:

```text
11 ff132bc23aaf5b56d00679a07961d23c2de30bafa7bbaa4e99ff966b61a49130
42 0623256abf3b068f212bacaa5c46f8bbcc6aabf2aeaa0f7ef5ecdc0ec8120cbe
73 2899df84afe4512ca127bf3367490a5bc42c88f3ad7868daf8195ed419ecb8fe
```

Cada ZIP do piloto contém oito arquivos: modelo, episódios de treino, avaliação, configuração/run, proveniência, commit, versões e manifesto. Os ZIPs das Fases 3 e 4 contêm episódios, resumo, configuração, manifesto, dois commits, execução e versões. O pacote da Fase 2 tem manifesto original mais restrito; não acrescentar metadados ao ZIP histórico. Dados complementares devem ser preservados ao lado dele.

Dependências diretas congeladas em `requirements-ci.txt`: Gymnasium 1.4.0, NumPy 2.3.5, pytest 9.0.2 e torch 2.10.0. Os runs oficiais usaram Python 3.11 e build CPU do PyTorch (`2.10.0+cpu`); o relatório operacional do piloto registra Python 3.11.17. `pip-freeze.txt` de cada artefato é a evidência das versões efetivamente instaladas, inclusive transitivas. Mesmo versões iguais não garantem reprodução bit a bit entre plataformas.

## O que foi demonstrado

1. Fase 1: três treinos independentes, 80 episódios cada, seguidos de 20 avaliações greedy e 20 aleatórias por checkpoint. As médias DQN 11/42/73 foram 10,35/15,10/13,40; as aleatórias, 0,35/0,20/0,20. O piloto não possui controles neurais não treinados nem causas de término explicitamente registradas.
2. Fase 2: 350 avaliações = 7 políticas × 50 seeds ambientais comuns 200001–200050, 10×10, 200 passos. Médias treinadas 10,08/13,40/12,38 contra não treinadas 0,00/0,08/0,14 e aleatória 0,26. Só três treinos independentes; episódio não é unidade de replicação do treinamento.
3. Fase 3: 1.050 avaliações = 7 políticas × 3 dimensões × 50 seeds 300001–300050. Os pesos treinados em 10×10 mantiveram vantagem sobre controles em 8×8 e 12×12. Não há equivalência demonstrada de score entre tamanhos nem transferência entre jogos.
4. Fase 4: 4.200 avaliações = 7 políticas × 12 estados × 50 seeds 400001–400050, 10×10, 200 passos. Médias treinadas 8,612/16,187/13,782; a não treinada 73 atingiu o limite em 598/600 episódios e coletou apenas 0,095 alimento em média. Limite de passos mede encerramento, não vitória.

## Limitações e decisões para a próxima fase

- Comparações de políticas dentro do mesmo estado compartilham a primeira comida física. A seed numérica comum entre estados/dimensões diferentes não garante comida, trajetória ou distribuição física equivalente. Contrastes entre estados não isolam causalmente orientação sem controles adicionais.
- Os 12 estados da Fase 4 estão relativamente centrais. Paredes/cantos, risco imediato e primeira comida controlada ainda precisam de protocolo novo e validação geométrica do corpo de três segmentos.
- O relatório 004 usou Welch entre dimensões; suas ressalvas sobre independência e distribuições devem acompanhar a interpretação, já que a mesma seed numérica reaparece. Uma futura análise deve explicitar a unidade de pareamento e a estrutura de correlação, sem reescrever os resultados históricos.
- CI e carregamento íntegro não demonstram eficácia. As inferências são condicionais aos três checkpoints, cenários e representação relativa 11D; não demonstram que uma única DQN aprende qualquer jogo.
- Recompensa original: +1 por alimento, −0,01 por passo sem alimento e −1 por colisão. Permanecer sem comida por 200 passos custa aproximadamente −2 na soma sem desconto registrada; uma colisão cedo pode custar menos nessa métrica. A DQN, porém, usa `gamma=0.98`: a sequência isolada de 200 penalizações tem retorno descontado de aproximadamente −0,4912, antes de considerar bootstrap no truncamento. Portanto a soma do CSV não basta para afirmar que o objetivo treinado favorece colisão. Isso motiva análise futura independente de recompensa, sem provar exploração intencional nem autorizar alterar os modelos existentes.
- Ausência de significância após Holm não comprova equivalência ou não inferioridade. Margens, tamanho amostral, famílias de testes e endpoint primário devem ser definidos antes de uma campanha nova.
- Os corpos das PRs e segmentos antigos de `STATUS.md` ainda registram campanhas como pendentes, embora comentários finais e APIs confirmem a conclusão. Tratar esses textos como registros históricos e publicar o estado consolidado atual. O roadmap original de `PROJECT_CONTEXT.md` usa numeração de fases diferente da sequência científica posterior; a sequência desta auditoria é a adotada nas PRs 1–4.
- Retomada exata de treinamento não é assegurada: checkpoints antigos não preservam replay e todos os estados de RNG. Não converter restauração de pesos em promessa de continuação determinística.

Preparar a Fase 5 e seus testes sintéticos está autorizado pela missão atual. A execução oficial depende de aprovação expressa do Chat 1 sobre protocolo exato, número de episódios, seeds, hipóteses, margem de robustez quando aplicável e SHA executável validado. Nenhuma aprovação histórica das Fases 1–4 autoriza repetição ou campanha da Fase 5.
