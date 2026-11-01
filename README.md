# NOVA Neural Arena 🐍

Laboratório **independente da IA NOVA** para testar aprendizado por reforço, começando com Snake e, futuramente, transferência entre ambientes. Resultados de Snake não demonstram aprendizado geral.

## Primeira experiência

- Snake com API Gymnasium, seeds e tabuleiro configuráveis.
- 11 entradas parciais (perigos relativos, direção, comida); três ações: seguir, virar à direita, virar à esquerda.
- DQN PyTorch com replay buffer, rede-alvo, política epsilon-greedy e checkpoint.
- Treino e avaliação em seeds distintas; comparação com política aleatória usando as mesmas seeds.
- Resultados: model.pt, episodes.csv, evaluation.json e run.json (versões, seeds, hiperparâmetros).

## Executar (Python 3.11+)

~~~bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m arena.training.train --episodes 100 --eval-episodes 20 --seed 42 --out experiments/local
~~~

Treino curto pode não melhorar a política aleatória. Avalie em múltiplas seeds e observe a variabilidade.

## GitHub Actions

Em **Actions**, selecione "Manual tests" ou "Manual short DQN train" e use **Run workflow**. Ambos exigem acionamento manual, e o treino limita episódios e publica os artefatos por 30 dias. Preserve modelos importantes fora desses artefatos temporários.

## Organização

- arena/envs: ambientes; arena/agents: políticas e redes.
- arena/training: treino/checkpoints; arena/evaluation: benchmarks.
- tests: testes; docs: fonte de verdade para decisões e estado.

Limitações: observação parcial, só Snake, CPU, nenhuma evidência de transferência. Checkpoint salva pesos/otimizador/contadores, mas não replay buffer nem RNG para retomada exata.
