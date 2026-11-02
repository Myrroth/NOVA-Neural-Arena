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

## Piloto controlado do Chat 3 (sem merge)

O novo workflow [`neural-pilot-on-label.yml`](.github/workflows/neural-pilot-on-label.yml) utiliza exclusivamente a ação **adicionar** a etiqueta `neural-pilot-approved` à **PR #1**. Nenhum push ou atualização de código inicia os treinos. O workflow roda três jobs independentes (seeds 11, 42, 73), cada um com 80 episódios de treino, 20 avaliações, Snake 10×10 e 200 passos máximos, sempre a partir do checkout explícito do SHA `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`.

Se a etiqueta não existir, crie-a em **Issues > Labels**; **aplicar a etiqueta à PR inicia treinamento real** e deve ser feito apenas pelo Chat 3 com a liberação do Planejamento. Após o run, baixe os três artifacts da aba Actions antes da expiração de 30 dias. Cada artifact contém modelo, métricas, versões e metadados de proveniência. Veja [`docs/STATUS.md`](docs/STATUS.md) para validação, pendências e alternativas. O workflow manual antigo `train.yml` continua sem alteração.

**Infraestrutura validada:** a etiqueta neutra `documentation` gerou execução `pull_request.labeled` com job ignorado, provando que uma etiqueta não autorizada não inicia o treinamento: https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41349938598. Após a verificação, ela foi retirada. O caso positivo `neural-pilot-approved` ainda não foi executado e depende do Chat 3.

## Fase 2 — preparação da avaliação dos checkpoints (sem treinamento)

Nova ferramenta em `arena/evaluation/frozen_phase2.py` e desenho detalhado em [`docs/PHASE2_PROTOCOL.md`](docs/PHASE2_PROTOCOL.md). A avaliação principal permanece **bloqueada** até o Chat 1 autorizar a etiqueta nova `neural-phase2-evaluation-approved` na PR da Fase 2. Modelos oficiais do run 41350959549 são validados por `SHA256SUMS`, SHA do checkpoint e proveniência; 3 DQN originais, 3 controles neurais **não treinados** e uma referência aleatória são preparados para a mesma sequência 200001..200050. Testes automáticos usam apenas modelos sintéticos e um microcenário; não executam a campanha. Os pesos são congelados e nenhuma rotina de aprendizagem é chamada.

A CI da PR #2 também validou em [Actions #41353674728](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353674728) os 3 artifacts oficiais, seus hashes e carregamento congelado **sem episódios**. O teste da etiqueta não autorizada foi [skipped](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353590209). Os workflows da Fase 2 selecionam somente testes que não treinam.


## Fase 3 — avaliação de generalização entre tamanhos (proposta, sem campanha)

A [documentação da Fase 3](docs/PHASE3_PROTOCOL.md) descreve o avaliador congelado `arena/evaluation/phase3_generalization.py`, os três tabuleiros 8×8/10×10/12×12 e 50 seeds de 300001 a 300050. Serão 7 políticas × 3 dimensões × 50 ambientes (1.050 partidas), **apenas após aprovação específica**. A CI da PR da Fase 3 usa checkpoints sintéticos para testes e confere apenas integridade dos checkpoints oficiais. `neural-phase3-generalization-approved` é uma etiqueta distinta, que não deve ser aplicada sem liberação do Chat 1. Snake, DQN e os resultados originais permanecem preservados.


**Validação da Fase 3:** avaliador congelado no commit `8892980eccfb623de30f3803a7550aa1fa31a9c6` ([CI original](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41369858499)); workflow de etiqueta usa checkout literal e compara `git rev-parse HEAD`. [CI reforçada](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370024631) com 24 testes aprovados e os três checkpoints oficiais verificados **sem campanha**; [teste de etiqueta neutra](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370045092) ignorado. O workflow real `phase3-on-label.yml` precisa de etiqueta autorizadora exclusiva **neural-phase3-generalization-approved** na PR #3 após decisão do Chat 1.


## Fase 4 — robustez da posição/orientação inicial (proposta técnica)

Consulte [docs/PHASE4_PROTOCOL.md](docs/PHASE4_PROTOCOL.md). A nova implementação `arena/evaluation/phase4_startstates.py` testa a variação isolada de cabeça (5,5)/(3,3)/(6,6) e orientação leste/sul/oeste/norte no Snake original 10×10. Protocolo futuro, sujeito a autorização: 7 políticas congeladas × 12 condições × 50 seeds ambientais 400001..400050 = **4.200 partidas de avaliação**, nenhuma realizada durante o desenvolvimento. A CI utiliza modelos sintéticos em testes minúsculos e audita os três checkpoints oficiais sem partidas. Nenhum treino/merge/autorização de etiqueta foi realizado.
