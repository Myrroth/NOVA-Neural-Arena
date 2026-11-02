# NNA-MULTIJOGOS-002 — protocolo finito candidato e análise offline

Este documento implementa o [pré-registro coordenador do Chat 1, comentário
6583974240](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6583974240).
É **candidato, não congelado e sem autorização de execução científica**.
O hash canônico produzido por `arena.universal.protocol` identifica conteúdo;
não concede permissão para treinar, avaliar, reservar ou repetir campanha.
Nenhum treino, jogo ou avaliação é executado pelo módulo de análise.

A missão mede cinco **simuladores locais preparados**, não os servidores reais
de Agar.io, jogos comerciais arbitrários ou inteligência universal. O legado
NNA-FECHAMENTO-001 e seus modelos não são alterados nem reaproveitados.

## Autoridade e divergências entre propostas

O comentário coordenador posterior do Chat 1 rege esta implementação: três
réplicas `1103,2207,3301`, GRU64, corrida como único jogo excluído, 60 mil
passos por jogo/réplica e bootstrap `510002`. As propostas anteriores de
[Chat 3](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6583916892)
e [Chat 4](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6583900244)
possuem budgets, seeds e folds diferentes; não se somam ao plano do Chat 1.
Não são autorização alternativa nem cinco folds executáveis.

Os detalhes analíticos abaixo que não eram literais no comentário original
são convenções candidatas explícitas e precisam de recertificação antes do
primeiro episódio científico. Depois disso, não alterar limiar, matriz,
recompensa, seed ou orçamento para buscar resultado positivo.

## Matriz e limites

Treino: fonte Snake/Pong/Agar/Flappy, `3 × 4 × 60.000 = 720.000` interações;
adaptação à corrida `180.000`; corrida do zero `180.000`. Total de treino:
**1.080.000** interações. Uma arquitetura/algoritmo compartilhados por réplica;
o comparador do zero tem estado próprio. Corrida é inteiramente excluída de
pré-treino, replay, normalizadores, modelo de mundo e escolha de configuração.

Uma única avaliação final, **1.735 episódios**, até 400 passos cada:

| stage | policy | jogos | réplicas | seeds por jogo/réplica | episódios | passos máximos |
|---|---|---:|---:|---:|---:|---:|
| before_adaptation | shared | 5 | 3 | 20 | 300 | 120.000 |
| after_adaptation | shared | 5 | 3 | 20 | 300 | 120.000 |
| from_scratch | scratch | corrida | 3 | 20 | 60 | 24.000 |
| baseline | untrained | 5 | 3 | 20 | 300 | 120.000 |
| baseline | random | 5 | 1, identificada por seed 0 | 20 | 100 | 40.000 |
| ablation | world_off | 5 | 3 | 20 | 300 | 120.000 |
| ablation | gru_reset | 5 | 3 | 20 | 300 | 120.000 |
| ablation | pixels_hidden | 5 | 3 | 5 | 75 | 30.000 |
| **Total** | | | | | **1.735** | **694.000** |

As seeds reservadas são `920001..920020`; pixels ocultados usam as cinco
primeiras. São proibidas em treino, debug, seleção de checkpoint e early
stopping. Antes/depois usam checkpoints predeterminados pelo budget, nunca o
melhor checkpoint no teste. Avaliação não faz gradientes nem insere replay.

Limites absolutos candidatos: 1.800.000 interações totais; 67.500 updates Q;
33.750 updates mundo; 101.250 combinados; 300 minutos CPU gratuita; RSS 3 GiB
por processo; uma tentativa e nenhum retry automático. O restante de 26.000
interações entre as matrizes e o teto não autoriza dados extras. O holdout do
modelo de mundo usa as próprias transições finais shared/after_adaptation,
disjuntas do treino; não exige coleta nem namespace adicionais. Preflight é
limitado a 2.000 passos aleatórios mecânicos e 100 updates sintéticos, sem
modelos científicos ou seeds finais. Viabilidade depende de medição real.
Falha/teto/resultado negativo preserva parciais e encerra; não há extensão.

`derive_training_seed` usa SHA256 de uma tupla JSON versionada, mantendo
treino fora do namespace final. Streams de ambiente de adaptação e scratch
são pareados; exploração e replay têm identidades próprias. O jogo não é
entrada da política; seus nomes servem a agendamento e auditoria.

## Dados e estimandos

`episodes.csv` contém somente os episódios finais. Colunas obrigatórias:
`model_seed,stage,policy,game,environment_seed,episode,steps,return,score` e
telemetria bruta do jogo. Duplicatas, células extras, seeds fora da matriz,
valores não finitos e episódio acima de 400 passos invalidam a matriz. Não
repor uma célula ausente após falha. `step_limit` não significa vitória.

Para cada contraste, a unidade independente é a **réplica treinada**, três
no total. Ambiente/episódio não vira um quarto agente. As diferenças são
pareadas por réplica e seed de ambiente. O bootstrap percentil tem 10.000
reamostragens: primeiro três réplicas com reposição; dentro de cada uma, as
seeds pareadas com reposição. RNG raiz `510002`, substream SHA256 pelo nome
estável do contraste. Reportam-se ganho, IC descritivo 95%, três diferenças
individuais e número de réplicas favoráveis. Três treinos oferecem incerteza
grande: não alegar robustez universal, poder confirmatório ou equivalência.

O random é uma campanha compartilhada entre réplicas, não três campanhas
aleatórias independentes. Para aprovar ganho, exigir margem pré-registrada,
limite inferior do IC estritamente maior que zero e pelo menos duas das três
réplicas favoráveis. Contra rede não treinada, a convenção candidata exige
ganho/IC positivos sem inventar uma segunda margem absoluta: Chat 1 deve
recertificá-la. Todos os contrastes/negativos são publicados; não escolher
o melhor jogo, seed ou checkpoint. Os gates são conjuntivos.

Taxa de intercepção é `Σhits/Σincoming`, não média sem peso das taxas dos
episódios. Colisão por distância também usa razões de somas. Sem denominador
positivo, a taxa é não estimável; nunca substituir por zero ou sucesso.

| requisito | endpoint e limite |
|---|---|
| 1 Snake | ganho de alimentos/episódio ≥1 vs random; controles não treinados; sucesso em pelo menos dois contextos de spawn; colisões de parede/corpo reportadas |
| 2 Pong | ganho de pontos líquidos ≥0,5 e intercepção ≥0,10 absoluta vs random; pontos, contatos e oportunidades brutos |
| 3 Agar | ganho de pellets ≥2; presas consumidas ≥0,3; redução de morte ≥0,10; exposição a maiores e menores presente; auditoria de cenários/ameaças |
| 4 Flappy | ganho de tubos realmente ultrapassados ≥1; sucesso em dois estratos de gaps/alturas |
| 5 Corrida | ganho de progresso de volta ≥0,15; conclusão ≥0,10 absoluta; colisão/distância não maior; auditoria de aceleração/freio/curvas |
| 6 Compartilhamento | receipts de pesos, arquitetura, replay e gradientes comuns, sem especialistas/hints |
| 7 Pixels/GRU | trocar pixels altera ≥5% de ações em 4/5 jogos; reset GRU altera ≥1% em 4/5; cada ablação reduz objetivo em ≥2 jogos |
| 8 Mundo | MSE latente ≤95% da persistência por jogo/réplica; MAE recompensa e Brier terminal reportados com suporte; mundo muda ≥1% de ações em 4/5 jogos e melhora objetivo em ≥2 |
| 9 Persistência | receipt de 25 transições/updates futuros reproduzidos; parâmetros/replay/RNG/estado/hidden/optimizer completos, incompatíveis recusados |
| 10 Transferência/retenção | corrida adaptada supera scratch por ≥0,05 volta, IC inferior >0 e 2/3 favoráveis; AUC treino ≥10% superior; cada réplica/jogo fonte retém ≥80% do ganho positivo anterior |
| 11 Separação | matriz completa, budgets/updates/holdout e ausência de seleção posterior auditados |
| 12 Entrega | demonstrações reais, proveniência, recuperação e upload/download Drive verificados; TeraBox pendente aceito |
| 13 Coordenação | aceites independentes Chat2/3/4 e decisão conclusiva Chat1 com fontes |

Ganho anterior de retenção não positivo torna a retenção relativa **não
estimável**. Reportar os ganhos antes/depois, sem divisão por zero. A
convenção candidata usa as primárias foods/net_points/foods/pipes_passed;
competências Agar e Pong adicionais continuam gates próprios, não são
substituídas pela retenção de um score agregado.

`training_curve.csv` guarda estágios `pretrain,adaptation,from_scratch`,
`model_seed,game,training_step,global_step,progress_normalized` onde aplicável.
Para corrida, `training_step` é local ao ramo (0..60.000). AUC candidata é a
soma do progresso normalizado de cada episódio multiplicado pela duração
até seu endpoint, dividida por 60.000. Incluir o episódio interrompido no
limite; não adicionar avaliação. Exigir fim exatamente em 60.000 e endpoints
estritamente crescentes. É curva de comportamento exploratório de treino,
não uma curva de testes independentes. AUC scratch zero não produz razão
infinita nem aprovação de transferência.

O gate de estratos Flappy usa os contadores físicos `narrow_pipes_passed` e
`wide_pipes_passed`: ganho positivo vs random em ambos, com IC/replicação do
método comum. O `gap_stratum` do próximo tubo ao terminar não rotula os tubos
já atravessados. Não atribuir todo score ao último gap observado.

## Atribuição e modelo aprendido

`action_pairs.csv` registra decisões contrafactuais **no mesmo estado e na
mesma trajetória compartilhada final**, não ações de episódios divergentes:
`model_seed,game,environment_seed,step,action_default,action_world_off,
action_gru_reset,action_pixels_hidden`. Há ablações separadas em episódios
para medir desempenho. Mudança de ação e melhora de objetivo são provas
distintas; nenhuma substitui a outra. Para pixels, usar as cinco seeds
previstas. Para GRU/mundo, todas as vinte. Inferências extras não permitem
interações extras nem treino. `info` não é entrada do controlador.

`model_predictions.csv` requer `model_seed,game,environment_seed,
transition_id,step,latent_mse_model,latent_mse_copy,reward_mae_model,
reward_mae_baseline,terminal_brier_model,terminal_brier_baseline,terminated`,
além de `stage=after_adaptation,policy=shared,baseline_source=training,
baseline_support`. Deve cobrir exatamente todas as transições das trajetórias
finais compartilhadas após adaptação. Não duplica episódios nem consome novos
passos. "Holdout" significa disjunção do TREINO, não disjunção das outras
métricas finais. Essa correção sucede a versão candidata 1 e precisa de
recertificação do Chat 1 antes de congelar o novo hash.
O mesmo alvo/encoder deve gerar ambos MSEs. Baselines e terminal base-rate
precisam vir somente do treino e ser registrados; não ajustá-los no holdout.
MSE de persistência zero ou suporte ausente não contam como aprovação.
Reportar eventos terminais raros e não confundir acurácia de “sempre vivo”
com previsão. O alvo é o hidden após observar o próximo RGB real com ação e
recompensa anterior clipada; baseline de persistência copia hidden atual.
Predição recompensa usa MAE em alvo clipado [-1,1], não Brier de variável
contínua; Brier refere-se somente à terminação Bernoulli. A análise exige
proveniência de treino dos baselines em receipt separado.
Predição auxiliar não é planejamento: uso causal do mundo tem gate próprio.

## Receipts, análise e estado conclusivo

`evidence.json` aponta a receipts reais para gates de engenharia/revisão.
Cada registro tem `status: "PASS"` e `receipts: [{"path": "arquivo-relativo",
"sha256": "64 caracteres hexadecimais"}]`. O analisador exige arquivo
existente dentro do diretório da campanha e hash correspondente. Um booleano
solto, constante de implementação ou receipt faltante nunca aprova gate.
O conteúdo dos receipts continua sujeito a revisão independente: hash prova
integridade de bytes, não a verdade de uma declaração.

Nomes: `shared_policy`, `observation_audit`, `world_holdout_provenance`,
`checkpoint_resume_25_updates`, `holdout_equal_budget`,
`separation_and_limits`, `agar_scenario_audit`, `racing_control_audit`,
`demonstration_recovery_delivery`, `coordinated_acceptance`.

`python -m arena.universal.analysis DIRETORIO --output analise.json` faz
somente leitura dos dados e escrita do relatório escolhido. Não importa
ambientes/treinadores nem executa jogos. `NOT_CONCLUDED` permanece enquanto
qualquer requisito falhar ou não tiver prova. Mesmo com todos os registros
aprovados, o estado `ALL_REQUIRED_EVIDENCE_REPORTED` **não emite aceite
da missão**; essa decisão pertence ao Chat 1 com revisão das fontes.

Os testes deste módulo usam números inventados, explicitamente fixtures.
Verificam matriz, aritmética, pareamento, denominadores, ausência de prova e
hashes; não comprovam aprendizagem, transferência ou competência real.

Pendências explícitas antes do congelamento: telemetria de volta/estratos,
oportunidades e timing auditáveis; proveniência dos baselines de mundo;
schema/receipts reais; convenções de AUC, comparação não treinada e retenção;
otimizador/perdas/target sync; identidades/hash de código/configs, CI e
viabilidade medida. Uma revisão candidata nunca transforma essas pendências
em provas. Nenhuma campanha é iniciada ou autorizada por estes arquivos.
