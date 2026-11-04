# Validação proposta da linhagem corrigida — NNA-CONTINUAL-003

**Estado: CANDIDATE_NOT_AUTHORIZED. Não congelado. Execução não autorizada.**

Este é um plano finito para avaliar o código de `arena/continual/`, desenvolvido na PR #14. Ele não inicia treinamento ou partidas, não concede autorização por existir em JSON e não reaproveita a autorização consumida da campanha 002. Os códigos, modelos, protocolos e resultados 001/002 permanecem separados. Os critérios abaixo não foram aprovados empiricamente pela versão corrigida: competência, retenção, transferência e benefício do mundo/memória/visão continuam **NOT_EVALUATED**.

A proposta usa os mesmos cinco simuladores locais: Snake, Pong, Agar, Flappy e Racing. A rede recebe somente RGB 32×32, sua própria ação anterior, recompensa anterior limitada a [-1,1] e a máscara mecânica constante das dez entradas. Não recebe ambiente, ID do jogo, telemetria privada, observações futuras ou estratégia vencedora específica de cada jogo. Isso define o alcance do teste; não estabelece capacidade de jogar qualquer jogo.

## O que será avaliado

São três réplicas independentes, com sementes de modelo **4103, 5207 e 6301**. Em cada réplica:

1. Treinar uma política compartilhada em Snake, Pong, Agar e Flappy, com **60.000 passos por jogo**, totalizando **240.000 passos-fonte**. Racing fica fora desta fase.
2. Preservar o checkpoint final dessa fase, o replay-fonte protegido e uma cópia congelada da própria política para destilação. Avaliar o checkpoint final-fonte nos cinco jogos, sem usá-lo para selecionar modelos ou modificar o plano.
3. Adaptar essa política a Racing por **60.000 passos**, mantendo os mecanismos de retenção previstos no código.
4. Treinar outro agente do zero em Racing por **60.000 passos**, com a mesma configuração e o fluxo ambiental alvo pareado com a adaptação.
5. Preservar os checkpoints finais exatos e executar a matriz abaixo uma única vez. Não escolher checkpoint por desempenho no teste.

O total proposto é **360.000 passos por réplica**, ou **1.080.000 passos de treinamento**. Episódios reservados para calibração entram nesses mesmos limites físicos; não acrescentam partidas. Os controles neurais sem treino usam a inicialização da respectiva semente, sem ajuste. O controle aleatório não usa rede e tem `model_seed=0`.

A configuração completa fica copiada literalmente de `asdict(ContinualConfig())` no protocolo, sem importar o treinador para apenas consultar o plano. Entre seus valores: contexto 16, unroll 8, retorno de três passos, lote de controle 16, lote de calibração 32, replay de ajuste 40.000, replay de calibração 8.000, intervalos Q/mundo 16/32, warmup 128, taxas de aprendizado 0,0003/0,001, gamma 0,98, fração alvo 0,5, peso de destilação 1 e mistura máxima 0,2. Alterações futuras nesses valores exigem nova revisão antes do congelamento.

## Matriz final, sem episódios adicionais

As sementes ambientais finais propostas são **930001–930020**. Sua presença neste documento e no plano não executa nenhuma delas. Os testes de engenharia novos usam ambientes artificiais e outras sementes.

| Etapa e política | Jogos | Réplicas × ambientes | Episódios |
| --- | --- | --- | ---: |
| Antes da adaptação, compartilhada | Cinco | 3 × 20 por jogo | 300 |
| Depois da adaptação, compartilhada | Cinco | 3 × 20 por jogo | 300 |
| Do zero, scratch | Racing | 3 × 20 | 60 |
| Baseline neural sem treino | Cinco | 3 × 20 por jogo | 300 |
| Baseline aleatório | Cinco | 1 × 20 por jogo | 100 |
| Ablação mundo desligado | Cinco | 3 × 20 por jogo | 300 |
| Ablação memória reiniciada | Cinco | 3 × 20 por jogo | 300 |
| Ablação pixels ocultos | Cinco | 3 × primeiros 5 por jogo | 75 |
| **Total** | | | **1.735** |

Cada episódio tem no máximo 400 passos. Logo, o teto dessa matriz é **694.000 passos de avaliação**. Treinamento mais avaliação somam no máximo **1.774.000 passos programados**, abaixo do teto global de **1.800.000**. A diferença não autoriza episódios extras, tentativas, preflights reais ou novo treinamento. Os máximos globais são **67.500 atualizações Q**, **33.750 atualizações do mundo** e **101.250 atualizações combinadas**.

As cinco demonstrações ficam fixadas antes de qualquer resultado: um episódio `after_adaptation/shared` de cada jogo, sempre com modelo **5207** e ambiente **930001**. Elas já fazem parte da matriz e serão gravadas na primeira passagem. Não haverá seleção dos melhores vídeos, regravação, substituição de um resultado ruim ou partida extra para completar demonstração ausente.

## Comparação justa e o que ela não demonstra

Adaptação e scratch recebem exatamente 60.000 interações físicas com Racing, a mesma configuração, a mesma agenda de oportunidades de atualização 16/32 e warmup de 128 passos por etapa. Seus ambientes usam a mesma derivação por episódio.

A quantidade efetivamente realizada de updates pode diferir. O replay precisa ter exemplos elegíveis e contexto completo; episódios reservados para calibração e durações que dependem das ações podem mudar quando isso ocorre. A decisão prévia é comparar **eficiência de amostras no mesmo orçamento ambiental**, sem afirmar igualdade de custo computacional. O relatório precisa publicar oportunidades programadas, tentativas, sucessos, skips de warmup/replay, skips por `InsufficientContext`, exposição física de ajuste/calibração e diferenças entre adaptação e scratch. Não se acrescentam updates para igualar os números, nem se interrompe uma execução válida apenas porque esses totais diferem. Uma exigência de igualdade estrita de updates demandaria outro desenho revisado antes do congelamento.

## Separação de fonte, ajuste, calibração e teste

O treinador corrigido usa o namespace `NNA-learning-corrections-v2`. As sementes de ajuste são **pares e ≥2.000.000**; as de calibração são **ímpares e ≥2.000.000**. Assim, ajuste e calibração são separados por paridade, e ambos ficam fora da faixa final 930001–930020. Essa separação vale também para as experiências dos jogos-fonte. O estágio-fonte real do treinador se chama `pretrain`; o protocolo aceita `source` como nome descritivo equivalente.

A derivação inclui modelo, jogo, episódio, domínio e papel ajuste/calibração. Adaptação e scratch usam intencionalmente o mesmo rótulo ambiental `target`, permitindo pareamento. O hash não prova ausência de colisões dentro de um fluxo finito; todas as sementes utilizadas devem ser registradas. As garantias estruturais são a separação de intervalos e a paridade. Nenhum dado final pode virar replay, calibração, critério de checkpoint ou origem dos baselines.

A cada dez episódios, um episódio inteiro fica reservado para calibração e é excluído das perdas de ajuste. A calibração utiliza somente o domínio ativo e transições distintas, identificadas por episódio e índice absoluto. Cada ação legal precisa de pelo menos oito transições válidas; todas as ações legais devem passar. O gate exige MSE de memória ≤0,90 da cópia, MAE de recompensa dentro da folga 0,02 e Brier de término dentro da folga 0,01 dos baselines locais. Sem suporte, em revisão antiga, em domínio novo ou com previsões ruins, a mistura fica zero. Atualizações do mundo invalidam a confiança antes de alterar pesos. Não se recalibra no teste final.

Esses critérios internos de calibração não substituem o critério científico de mundo no teste, cujo limite histórico continua sendo 0,95. Os dois têm papéis distintos.

## Contrafactuais, previsão e fronteiras da memória

Episódios de ablação têm seu próprio histórico e usam suas próprias ações e recompensas anteriores. Para atribuição de ações, há também pares no **mesmo estado corrente** em cada passo real da política compartilhada depois da adaptação: o controlador é copiado antes de observar o quadro; mundo desligado, GRU reiniciada e pixels ocultos são comparados sem alterar a trajetória principal nem executar passos ambientais adicionais.

A ablação da GRU muda somente a memória de entrada. Ela segue o mesmo gate e a mesma fase periódica; não desliga o mundo como efeito adicional. O planejamento normal fica desligado quando a observação seguinte cruzaria um reset periódico. Os registros publicam `planning_weight` e `context_reset`. O campo `q_mixed` é exatamente o resultado do controlador. Valores brutos do modelo podem aparecer no log mesmo com peso zero, mas não entram na ação por causa desse registro.

A previsão condicionada à ação escolhida é calculada e copiada **antes de `env.step`**. O alvo posterior é a continuação de um passo da GRU a partir da memória atual, observando o RGB seguinte real com a ação executada e a recompensa limitada. Seu rótulo é `target_mode="continuous_gru_one_step"`. O campo `next_context_boundary` informa se o próximo controle normal reiniciaria a memória. Nessa fronteira, o alvo contínuo de auditoria **não é apresentado como estado do próximo controle reiniciado**. O ajuste latente exclui sucessores que reiniciam contexto; a auditoria registra todas as transições e permite identificar as fronteiras explicitamente.

O erro de memória é comparado com copiar a memória atual. A recompensa contínua usa MAE contra a média calculada no **treinamento**; término usa Brier contra a taxa de término do **treinamento**. Truncamento por limite de tempo não é morte. Esses baselines nunca são estimados nas trajetórias finais. Previsões e comparações usam as trajetórias já previstas na matriz, sem novas interações.

## Mesmas margens científicas

As margens são copiadas de `arena/universal/protocol.py`; o código histórico permanece intacto. Contra o baseline aleatório, a política final compartilhada precisa atingir:

| Jogo | Ganhos exigidos |
| --- | --- |
| Snake | Alimentos: +1,0 |
| Pong | Pontos líquidos: +0,5; interceptação: +0,10 |
| Agar | Alimentos: +2,0; presas: +0,3; redução de morte: 0,10 |
| Flappy | Tubos: +1,0; ganho positivo nas travessias físicas estreitas e largas |
| Racing | Progresso: +0,15; conclusão: +0,10; colisões/distância não superiores ao random em nenhuma réplica |

Contra o controle neural sem treino, exige-se ganho pareado positivo com intervalo positivo; a margem adicional é zero. A exposição a oponentes maiores e menores em Agar precisa existir em cada réplica, mas proximidade não comprova por si só uma estratégia causal de fuga ou perseguição. O teste de Pong se limita ao oponente local assimétrico existente, sem alegar domínio geral de Pong.

Para cada contraste, as métricas por réplica são a soma dos numeradores dividida pela soma dos denominadores. Proporções agrupam oportunidades; não são médias simples das proporções de episódios. O ganho é a média dos três ganhos pareados.

São **10.000 reamostragens hierárquicas pareadas**: primeiro três índices de modelos com reposição; depois índices de ambientes com reposição em cada posição de modelo sorteada. Tratamento e controle usam exatamente os mesmos índices. Os controles aleatórios são compartilhados entre réplicas; não viram três agentes treinados independentes. O RNG da análise é local, com semente **530003 XOR SHA256 do rótulo do contraste**. O intervalo de 95% usa os percentis 2,5% e 97,5% da média dos ganhos reamostrados.

Aprovação exige simultaneamente ganho médio ≥ margem, limite inferior do intervalo >0 e pelo menos duas réplicas com ganho >0. Três agentes são evidência descritiva sob este protocolo, não garantia universal.

Outras exigências continuam conjuntivas:

- **Retenção:** em cada jogo-fonte e cada réplica, `(pós − random)/(pré − random) ≥0,80`. Pré-ganho não positivo produz evidência ausente, nunca aprovação.
- **Transferência final:** ganho pareado de progresso de Racing sobre scratch ≥0,05, com o mesmo intervalo positivo e duas réplicas favoráveis.
- **Transferência durante treino:** AUC do progresso ponderada pelos intervalos de passos, usando os pontos finais dos episódios, inclusive o parcial no limite exato de 60.000 passos. Exige média de `(AUC_adaptação/AUC_scratch −1) ≥0,10` e duas réplicas positivas. AUC scratch zero impede estimativa.
- **Mundo:** MSE/cópia ≤0,95 em cada jogo e réplica. MSE da cópia zero impede estimativa. MAE de recompensa e Brier de término são publicados com seus baselines, sem inventar uma margem mais fácil depois de observar resultados.
- **Ablações:** alteração de ações em pelo menos quatro jogos e ganho significativo no objetivo em pelo menos dois. As frações mínimas de ação alterada permanecem 0,05 para pixels e 0,01 para memória e mundo.
- **Persistência técnica:** continuidade de 25 transições/updates sintéticos com tolerância 1e-6. Essa prova técnica não substitui os critérios de aprendizado.

Falha, dado ausente, saída incompleta ou limite atingido não se transformam em PASS. Não há escolha de vencedor entre critérios, exclusão posterior de jogo ou redução das margens para aprovar o projeto.

## Orçamento candidato e pendências antes de congelar

O teto de CPU proposto é **300 minutos**, com parada interna aos **285 minutos** e memória máxima de **3 GiB**, contabilizando de forma cumulativa tanto tempo de processo quanto tempo de execução. Esses valores são **candidatos ainda sem aceite de viabilidade**. Não se presume que o perfil antigo represente esta arquitetura, os dois otimizadores, os forks de avaliação, a calibração ou os registros adicionais.

Antes do congelamento, falta:

1. Medir o perfil **sintético** da nova versão, projetar os custos de treino, matriz e registros, incluir sobrecargas e verificar o encaixe nos limites gratuitos de CPU/memória. Se a projeção não couber, rever o desenho antes de qualquer autorização; não iniciar esperando ultrapassar o teto.
2. Receber revisão técnica/científica independente, incluindo separação dos dados, retenção, semântica das fronteiras da GRU, contadores de updates e critérios históricos. A solicitação mais recente ao Chat 1 não é resposta nem recertificação; nenhuma aprovação dele está registrada neste candidato.
3. Fixar fonte e fluxo de execução novos por commits/hashes imutáveis. Verificar reserva única, concorrência global sem cancelamento, limites cumulativos, preservação de artefatos, ausência de retries e ausência de seleção por teste.
4. Preparar a solicitação de **consentimento humano novo e específico** para o protocolo final congelado, fonte, fluxo, custos e limites. Esse consentimento precisa ser recebido antes da execução; o arquivo do plano, aprovações de engenharia ou autorizações antigas não o fornecem.

`protocol_sha256()` identifica o conteúdo candidato para revisão. Calcular esse hash não congela o protocolo e não autoriza a campanha. `python -m arena.continual.protocol` somente descreve o plano, usando a biblioteca padrão; não importa Torch/Gym nem cria modelos ou ambientes.

## Encerramento definitivo da tentativa autorizada futura

Se e quando todos os requisitos anteriores estiverem resolvidos, o desenho permite **uma única tentativa finita**, sem merge, recurso pago, repetição automática ou novo treino após resultado negativo. A execução termina nos checkpoints finais e na matriz definida, ou para antes por falha/limite, preservando tudo o que já produziu. Uma parada não autoriza reiniciar ou completar episódios faltantes em outra tentativa.

O pacote final deve conter protocolo e hashes, fontes/fluxo fixados, prova da autorização e da reserva, checkpoints finais disponíveis, curva completa até os limites atingidos, contadores de passos/updates/exposição, resultados brutos por célula, pares no mesmo estado, previsões e baselines, as cinco demonstrações disponíveis e análise de todos os critérios. Resultados negativos ou incompletos ficam publicados como tais. Depois dessa tentativa, o resultado científico encerra esta campanha; qualquer proposta posterior seria outra decisão explícita, sem loop automático de treinamento.
