# NNA-MULTIJOGOS-002 — implementação da visão completa

Missão ativa: [Issue12](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12).
O fechamento anterior consolidou Snake e preparou Pong; não implementou a visão
inteira. Esta linhagem nova mantém todos os modelos e executáveis históricos
intactos. Nenhum resultado da missão anterior é substituído por uma alegação
de aprendizagem multijogos.

## Critério de conclusão

Os cinco jogos precisam de aprendizagem medida em dados reservados: coleta e
sobrevivência em Snake; intercepção e pontos em Pong; coleta, fuga e consumo de
menores no Agar-like; saltos/passagem de obstáculos em Flappy; velocidade,
curvas e colisões em corrida. Também são obrigatórios: um algoritmo/política
compartilhados, percepção comum, memória temporal/persistente, previsão
aprendida de consequências e adaptação/transferência para um jogo inteiro
excluído do pré-treino, frente a inicialização nova com orçamento igual.

Uma CI verde, um simulador, uma demonstração aleatória ou um arquivo de pesos
não comprovam esses resultados. Critérios quantitativos, seeds, estimandos e
orçamentos serão registrados antes da avaliação oficial. Não inferir domínio
universal de jogos desconhecidos a partir de simuladores preparados.

## Contrato visual e controles

`arena.universal.envs` contém cinco simuladores locais novos: VisualSnake,
VisualPong, AgarLike, FlappyBird e Racing. São outra linhagem de ambiente,
com IDs `visual-<jogo>-v1`, sem modificar os ambientes históricos. Observações
são imagens RGB uint8 de 32×32 pixels. Não são fornecidas features de perigo,
distâncias a objetivos, trajetórias ideais ou ações recomendadas ao agente.

O controle é o mesmo joypad Discrete(10): parado, cima, baixo, esquerda,
direita, cima-esquerda, cima-direita, baixo-esquerda, baixo-direita, botão.
Pong usa cima/baixo, Flappy usa botão; corrida usa direções para esterço,
aceleração e freio, inclusive combinações. Um botão sem ligação é ignorado.
A máscara informa somente quais controles existem, nunca quais são seguros.

O estado completo/metadados/métricas são usados pelo simulador, snapshots e
análise; a política deve receber apenas pixels, máscara e sua própria história
de ações/recompensas. Adversários programados pertencem ao ambiente, não são
o agente aprendiz e nunca fornecem demonstrações ao treinamento.

As recompensas são objetivos de jogo, não políticas: comida/colisão no Snake;
retorno de bola, ponto e ponto sofrido no Pong; alimento/consumo/morte no Agar;
passagem/sobrevivência/colisão no Flappy; progresso/custo temporal/saída da pista
na corrida. Metas e métricas brutas permanecem distintas das recompensas.

## Arquitetura em implementação

Uma CNN pequena sobre pixels, memória GRU, um único head de valores para as
dez ações e Double DQN recorrente. Sem identificador ou head por jogo.
Replay de sequências balanceado entre ambientes, separado por limites de
episódio. Modelo de mundo condicionado à ação prediz próxima representação,
recompensa e término; sua utilização e fidelidade precisam de avaliação real.
Um componente apenas auxiliar não será apresentado como planejamento.

Snapshot completo precisa conservar pesos online/target, modelo de mundo,
otimizador, replay, RNGs, contadores, estados dos ambientes, observações,
ações/recompensas anteriores e memória recorrente. Retomar deve reproduzir
ações/amostras/atualizações no mesmo código e ambiente de dependências.
Essa é uma propriedade nova; não é atribuída aos checkpoints históricos.

## Estado verificado inicial

- Cinco simuladores implementados com pixels, física, métricas e snapshots.
- 21 testes locais aprovados: contrato oficial Gymnasium, sementes, resultados
  físicos, alimentação por tamanho, colisão, salto, throttle/freio e retomada.
- 500 transições aleatórias de smoke técnico concluídas; nenhum treinamento,
  nenhuma conclusão de competência e nenhum dado científico de aprendizagem.
- Agente visual/treinador/memória e avaliação ainda em implementação.

Os dados dos testes são fixtures; não entram na avaliação oficial. O protocolo
e a execução científica nova dependem de código/CI/SHA fixos e liberação
específica do Planejamento. As autorizações e etiquetas antigas não se aplicam.
Sem merge, gasto pago ou agenda recorrente. TeraBox continua pendente por
decisão do usuário; Drive ATUAL anterior é preservado até substituição validada.
