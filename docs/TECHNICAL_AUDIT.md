# Auditoria técnica do laboratório

Data local: 2026-11-01 (America/Sao_Paulo). Base auditada:
`af4bd307c6b6a75980a1b66c007cfd6a1d9a5d59`, branch da PR #4.
Esta revisão acrescenta arquivos novos e conserva os avaliadores históricos,
Snake, DQN, treinador, workflows históricos e pesos originais.

## Resultado e limites da auditoria

Foram inspecionados Snake/Gymnasium, DQN, treinamento, benchmark, avaliadores
das Fases 2–4, testes e workflows presentes nessa base. Os resultados históricos
não foram recalculados. Uma vulnerabilidade de interface não demonstra que os
artefatos oficiais foram adulterados: os workflows históricos verificam seus
hashes imediatamente antes do carregamento e fixam commits de avaliação.

| Achado demonstrável | Consequência | Tratamento nesta revisão |
|---|---|---|
| `load_verified_model(..., verified=...)` histórico aceita um dicionário fornecido pelo chamador e reabre `model.pt` sem verificar os bytes novamente | Um chamador pode pular a verificação; uma troca do arquivo entre verificação e carregamento também pode passar despercebida | Novo `load_official_checkpoint` não aceita `verified` nem hash substituto. Revalida a proveniência e carrega um snapshot dos bytes cujo hash oficial acabou de conferir |
| O carregador histórico confere compatibilidade dimensional, mas não rejeita explicitamente tensores NaN/Inf, dtype convertido ou dimensão oculta excessiva | Um checkpoint incorreto pode carregar ou alocar um modelo desnecessariamente grande | Novo carregador exige campos/tipos/limites da configuração, nomes/shapes/dtypes exatos, tensores CPU finitos, hash confiável e tamanho do arquivo limitado |
| A construção de `QNetwork` durante o carregamento histórico avança o RNG global de PyTorch | Chamadas futuras que dependam da sequência global podem receber inicializações diferentes | Novo carregador usa `torch.random.fork_rng` durante a construção e testa a preservação do RNG |
| Treinador e avaliadores históricos abrem saídas com `w` e diretórios existentes com `exist_ok=True` | Reutilizar a mesma pasta pode substituir métricas/checkpoints silenciosamente | `write_new_json` usa criação exclusiva; a infraestrutura nova deve reservar uma pasta nova antes de qualquer avaliação. APIs congeladas permanecem históricas |
| Checkpoint de `DQNAgent.save` contém redes/otimizador/contadores, mas não replay nem estados dos RNGs | Carregamento permite inferência e continuação aproximada; não é retomada bit a bit do treinamento | Limitação documentada. Um formato de retomada completa exigirá experimento e versão próprios |
| `DQNAgent(config, seed)` sem `torch.manual_seed` externo controla ações/replay, mas não a inicialização neural global | Uso isolado da API não garante pesos idênticos para a mesma seed | O treinador oficial semeia PyTorch antes da criação; preservado. Novos controles usam o helper de inicialização isolada |
| Versões de execução são fixas em `requirements-ci.txt`, enquanto dependências de distribuição são intervalos | Instalar apenas o pacote hoje pode produzir ambiente diferente do oficial | Recuperar com o freeze e versões do artefato; executar CI com constraints. Não alegar reprodução bit a bit entre OS/versões diferentes |
| Workflows históricos usam tags versionadas de Actions e artifacts de 30 dias | Código da Action pode mudar sob uma tag; downloads oficiais expiram | Preservação externa verificada e catálogo de proveniência nesta revisão. SHA completo das Actions e mecanismo de recuperação durável precisam de revisão antes de campanhas futuras |
| Documentos congelados de Fases 2–4 ainda descrevem campanhas como pendentes | Leitor pode confundir preparação antiga com situação atual | Consultar `HISTORICAL_AUDIT.md` e fontes oficiais atuais. Textos históricos continuam preservados como registro de sua época |

## Contrato do ambiente e DQN

- `SnakeEnv.reset` semeia o gerador Gymnasium; comida é escolhida uniformemente
  entre células livres em ordem fixa. Ações são relativas: seguir, direita,
  esquerda. O passo distingue colisão/preenchimento (`terminated`) de limite de
  passos (`truncated`). Mover para a cauda liberada é permitido se não houver
  alimentação. O corpo inicial padrão tem três segmentos.
- A observação de 11 atributos contém perigo nas próximas três ações, direção e
  sinais da posição relativa da comida. Não contém corpo completo, distâncias ou
  histórico; é parcialmente observável. Repetir a observação 11D não prova ciclo
  do estado físico. Instrumentação futura deve comparar corpo ordenado completo,
  direção e comida; passos restantes afetam o horizonte e devem ser registrados.
- DQN usa replay, target network, Huber, clipping de gradientes e epsilon
  decrescente. Não é Double DQN: o mesmo target escolhe e avalia o máximo. Essa é
  uma escolha do algoritmo original, não um erro a corrigir nos pesos congelados.
- O treinador armazena apenas `terminated` como terminal para o bootstrap.
  Isso é coerente quando o limite de 200 passos é um truncamento externo de uma
  tarefa contínua. Se uma fase futura definir horizonte finito intrínseco, o
  objetivo e a observação de tempo deverão ser revistos em experimento separado.
- Seeds de inicialização oficiais são 11, 42, 73. As faixas de seeds ambientais
  de treino se sobrepõem entre runs; não tratá-las como datasets disjuntos.
  As seeds de avaliação mantidas por fase são distintas das do treinamento.

## Recompensas: interpretar antes de alterar

O código histórico dá `+1` ao comer, `-0.01` por passo comum e `-1` na colisão.
Com `F` alimentos, `S` passos e indicador `C` de colisão, o total não descontado
é `F - 0.01*(S-F-C) - C = 1.01F - 0.01S - 0.99C`.

Portanto, reward, score e sobrevivência medem coisas diferentes. Sem alimentos,
200 passos seguros somam `-2`, enquanto colisão imediata soma `-1`; a reward
reportada pode ranquear o segundo acima do primeiro. Isso **não** demonstra que
o objetivo DQN favoreça colisão: com `gamma=0.98`, a soma descontada de 200 passos
comuns é aproximadamente `-0.491`, maior que `-1` da colisão imediata. A política
também depende das estimativas aprendidas e do bootstrap. Não alterar recompensas
ou reinterpretar resultados históricos como se tivessem outro objetivo.

Manter alimentos como desfecho primário. Investigações futuras independentes
podem variar penalidade por passo, horizonte e desconto, comparando alimentos,
tempo até primeira coleta, eficiência e ciclos, com novos treinos e controles.

## Reprodutibilidade e integridade novas

`arena/shared/integrity.py` fornece SHA-256, leitura limitada verificada, caminhos
relativos seguros, verificação estrita de manifests e JSON sem substituição.
Manifest interno detecta corrupção; sozinho não autentica a origem. O digest do
ZIP ou modelo deve vir de catálogo confiável independente, não do próprio arquivo.
Arquivos adicionais não listados em um manifest não são autenticados por ele.

`arena/shared/checkpoints.py` nunca instancia agente ou otimizador, não faz
inferência e usa somente `torch.load(..., map_location='cpu', weights_only=True)`.
Não existe fallback para pickle irrestrito. O carregamento genérico aceita hash
explícito para fixtures ou outros modelos cuja identidade já foi estabelecida;
o carregamento oficial usa a tabela congelada do piloto e configuração exata.
Limites e `weights_only` não transformam PyTorch em sandbox: só carregar bytes
cujo digest tenha uma origem confiável. Esta revisão não promete segurança de
desserialização para checkpoints arbitrários obtidos da internet.

Os testes novos não jogam nem treinam. Cobrem digest incorreto antes de
desserializar, troca do arquivo após snapshot, RNG preservado, ausência de
otimizador, schema/NaN/dtype/shape, caminhos inseguros, manifest duplicado,
adulteração e recusa de substituição de JSON. Resultado executado da suíte é
registrado no relatório de entrega/CI, não inferido a partir da compilação.

Validação local efetiva, sem dependências externas: `compileall` dos dois módulos
e do arquivo de testes passou. Um smoke de biblioteca padrão executou e aprovou
11 rejeições de caminhos inseguros, manifest aninhado com conjunto exato, leitura
com hash confiável, rejeição de hash incorreto/limite de bytes/manifest duplicado
e recusa de substituir JSON. A suíte de checkpoint/PyTorch requer a instalação
das dependências oficiais e a evidência da CI; não foi substituída por esse smoke.

A [CI inicial 41377237661](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41377237661)
posteriormente confirmou **140 testes aprovados**, incluindo todos os testes de
checkpoint/PyTorch desta revisão. Os dois avisos informativos vêm do verificador
de renderização do Gymnasium, sem falhas. As Actions dos workflows novos agora
estão fixadas por SHA completo; as tags dos workflows históricos foram preservadas.

## Qualidade científica e próximas decisões

1. **Unidade de inferência:** três checkpoints representam três treinos, mesmo
   quando cada um tem centenas de episódios. Intervalos por episódio descrevem
   incerteza ambiental condicional a esses pesos, não variabilidade de futuros
   treinamentos. Não misturar essas unidades em um único erro padrão.
2. **Pareamento:** exige chave de cenário e seed iguais e definição explícita
   do alimento inicial. Igual seed entre corpos/tamanhos diferentes não garante
   comida física igual. Depois da primeira comida, estados e conjuntos de células
   livres divergem; RNG compartilhado não iguala trajetórias.
3. **Primeira comida:** ausência de coleta é dado censurado/falha de coleta, não
   valor zero de tempo. A média apenas entre quem comeu tem viés de seleção;
   reportar taxa de coleta e denominadores junto, além dos tempos condicionais.
4. **Ciclos:** detecção deve ser passiva e não terminar o episódio. Mesmo estado
   completo repetido indica recorrência; para política determinística sem sorteio
   intermediário, pode sustentar diagnóstico de ciclo. Política aleatória pode
   sair dele, e `step_limit` nunca equivale a vitória.
5. **Incerteza:** pré-registrar contrastes, unidade amostral e procedimento de IC
   pareado por seed. Uma opção a validar é bootstrap por blocos de seed, mantendo
   todos os cenários/políticas do bloco juntos. Não tratar células correlacionadas
   como amostras independentes. Uma família de testes deve ter correção conjunta,
   por exemplo Holm, definida antes da campanha.
6. **Não inferioridade/equivalência:** falta de significância não prova nenhuma
   delas. Chat 1 deve escolher margem em alimentos, justificar poder/tamanho da
   amostra e aprovar novo SHA/protocolo. Nenhuma margem foi inferida retroativamente.
7. **Extensão para Pong:** separar adaptador de observação/ação, configuração,
   proveniência e análise das especificidades do Snake. Um checkpoint de entrada
   11D e três ações do Snake não tem competência presumida em outro jogo. Criar
   ambiente testado e novos controles antes de experimentar transferência.

Nenhuma decisão acima autoriza campanha, novo treinamento, merge ou modificação
da NOVA original. A Fase 5 permanece infraestrutura em revisão.
