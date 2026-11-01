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

## Validação reprodutível de desenvolvimento

O CI utiliza Python 3.11 e versões fixadas das dependências principais em `requirements-ci.txt` (Gymnasium 1.4.0, CPU PyTorch 2.10.0, NumPy 2.3.5, pytest 9.0.2). Para espelhar o ambiente de CI:

~~~bash
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-ci.txt -e '.[dev]'
python -m pytest -q
~~~

O primeiro CI oficial passou em 2026-11-01 (13 testes, 2 avisos informativos relacionados a `env.spec` em instanciação direta): https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41330556392. A suíte inclui `env_checker` oficial em modo normal e ANSI, além de teste de repetição de métricas com seed fixa. Versões transitivas não estão completamente travadas; CI registra `pip freeze` para auditoria. Checkpoints não permitem retomada exata (replay/RNG não persistidos).

## GitHub Actions

Para validar a PR sem alterar `main`, o workflow de **testes** também roda em eventos de `pull_request`; **treinamento continua exclusivamente manual** (`Manual short DQN train`). `workflow_dispatch` só fica disponível quando o workflow estiver na branch padrão; até lá, não presumir que o botão manual funciona. O treino limita episódios e publica os artefatos por 30 dias. Preserve modelos importantes fora desses artefatos temporários.

## Organização

- arena/envs: ambientes; arena/agents: políticas e redes.
- arena/training: treino/checkpoints; arena/evaluation: benchmarks.
- tests: testes; docs: fonte de verdade para decisões e estado.

Limitações: observação parcial, só Snake, CPU, nenhuma evidência de transferência. Checkpoint salva pesos/otimizador/contadores, mas não replay buffer nem RNG para retomada exata.
