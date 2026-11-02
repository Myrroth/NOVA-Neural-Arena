# DECISIONS — NOVA Neural Arena

## 2026-11-01 — Laboratório separado
Repositório independente da IA NOVA; integração apenas depois de validação e autorização explícita.

## 2026-11-01 — Ordem dos ambientes
Snake, depois Pong, depois simulador Agar-like local. Motivo: aumentar dificuldade gradualmente.

## 2026-11-01 — Fonte de verdade
PROJECT_CONTEXT.md, STATUS.md e DECISIONS.md versionados; conversas são auxiliares.

## 2026-11-01 — Stack inicial
Python 3.11+, Gymnasium para API, NumPy para observações, PyTorch para DQN, pytest para testes. DQN com rede-alvo, replay e epsilon-greedy; observações parciais de 11 elementos e 3 ações relativas. Preferir simplicidade para CPU e benchmarks reproduzíveis.

## 2026-11-01 — Validação e automação
Seeds de avaliação separadas (seed + 100000); DQN greedy versus baseline aleatória nas mesmas seeds; relatório contém média, mediana, desvio, mínimos e máximos. Actions somente por acionamento manual, limites explícitos, permissões read-only e artifacts com retenção de 30 dias. Resultados de treino curto não equivalem a generalização. Checkpoint permite inferência; replay e estados de RNG ainda não são persistidos.

## 2026-11-01 — Validação técnica da PR #1
Adotar Gymnasium oficial 1.4.0 no CI e pin de dependências diretas (requirements-ci.txt); registrar dependências resolvidas com pip freeze. O CI de testes aceita evento de pull_request, necessário para testar antes de merge, enquanto treinamento permanece exclusivamente via workflow_dispatch. Testes de contrato usam env_checker real, sem fallback/skip. A impossibilidade de instalar dependências na máquina de verificação local não deve ser relatada como aprovação oficial. Fixar algoritmos determinísticos PyTorch (CPU, thread única) e comparar métricas de execuções curtas com mesma seed.

## 2026-11-01 — Resultado da validação do Gymnasium e Actions
Workflow #41330556392 completou com 13 testes aprovados e env_checker real Gymnasium 1.4.0 aprovado para modos None e ansi. Dois avisos sobre ausência de env.spec com instanciação direta não constituem falha, já que modos foram verificados explicitamente. Os antigos actions/checkout@v4, setup-python@v5, upload-artifact@v4 geraram avisos de Node20; migramos para v7/Node24, mantendo testes via pull_request e treino somente workflow_dispatch. Não há validação de treino extenso neste Chat 2.

## 2026-11-01 — Piloto opt-in por etiqueta na PR #1
Decisão do Chat 1 (issuecomment-6563787807): desbloquear o treinamento de três seeds por infraestrutura **sem** modificar o commit de código principal validado. Adicionado workflow `neural-pilot-on-label.yml` com evento exclusivo `pull_request.labeled`, job condicionado à etiqueta `neural-pilot-approved`, PR #1 e origem no próprio repositório. Checkout imutável do commit `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`; matriz de seeds 11/42/73 e protocolo 80+20/seed, 10×10, 200 passos. `pull_request_target` rejeitado por risco de executar código não confiável. Treino não ocorre em push ou sincronização da PR. Artifacts separados, limitados a 30 dias, com métricas, modelo, freeze, proveniência e hashes. **Disparo exclusivamente pelo Chat 3 autorizado a aplicar a etiqueta.**

## 2026-11-01 — Prova do evento sem executar o piloto
Teste de gatilho com etiqueta neutra `documentation` aplicada temporariamente à PR #1. Workflow GitHub Actions https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41349938598 foi iniciado com evento `pull_request` e terminou `skipped`, conforme condição de não autorização. Etiqueta removida depois. Nenhum treino foi iniciado. CI da PR após a alteração aprovada em https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41349905467. O caso positivo somente poderá ser disparado pelo Chat 3 aplicando `neural-pilot-approved`.

## 2026-11-01 — Fase 2: isolamento da avaliação sem aprendizagem
Nova PR baseada na branch da PR #1. Avaliar somente os checkpoints oficiais e verificados do run 41350959549 em seeds ambientais comuns 200001..200050, com versões não treinadas da mesma QNetwork (inicializações 11, 42 e 73) e política aleatória fixamente sem treino. Introduzir registro de desfechos na camada nova, sem alterar o simulador original. Avaliação completa proibida até aplicação da etiqueta distinta `neural-phase2-evaluation-approved` após novo aceite do Chat 1. Unidade/CI devem usar apenas checkpoints sintéticos e subconjunto mínimo. Dependências oficiais, sem merge, sem novos pesos e preservar a origem SHA 24bd38e9e78a14e894f32d7d1c5b5b6d6d724463.

## 2026-11-01 — Integridade oficial comprovada sem avaliação
Na PR #2, CI https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353674728 baixou os 3 artifacts originais, conferiu 7 hashes por seed, SHA fixo/proveniência e carregou QNetwork congelada sem episódios. O evento de etiqueta neutra (run 41353590209) foi ignorado. Por contenção do escopo sem treinamento, restringimos workflows da Fase 2 apenas aos testes do Snake e ao conjunto de testes sintéticos da Fase 2, excluindo testes legados do trainer. A campanha de 50 seeds exige autorização posterior do Chat 1.

## 2026-11-01 — Fixar o commit imutável do avaliador (Chat 1 / PR #2)
Decisão do Chat 1 issuecomment-6565513811: o commit de avaliação validado `f9e1b171a8c045f8bd7c874432114e4bc654665e` deve ser usado literalmente no checkout da campanha, em vez do SHA mutável do HEAD da PR. Verificação independente com `EXPECTED_HEAD` e `git rev-parse HEAD`; novo teste estático detecta regressões e confirma que só `pull_request.labeled` com `neural-phase2-evaluation-approved` poderá acionar a campanha. Manter integralmente proteção de integridade dos checkpoints, modelos congelados e ausência de treinamento. **Nenhuma autorização de execução de 350 episódios.**


## 2026-11-01 — Fase 3: OOD somente por dimensões, sem retreino
Decisão do Chat 1 na PR #2 (issuecomment-6566397649): nova PR isolada baseada na Fase 2 para avaliar, **após autorização futura**, 3 redes treinadas oficiais, 3 redes não treinadas com as mesmas inicializações e 1 aleatória em 8×8/10×10/12×12, 50 seeds ambientais 300001..300050, 200 passos (1.050 episódios). Criar indicadores de primeira coleta e alimento por 100 passos na camada de avaliação, sem alterar Snake, recompensas, direção inicial, modelos ou as fases anteriores. Testes pequenos com checkpoints sintéticos. Workflow exclusivo `pull_request.labeled` e etiqueta nova `neural-phase3-generalization-approved`, com checkout fixado em SHA auditado após CI. Nenhuma campanha nesta preparação.


## 2026-11-01 — Avaliador Fase 3 congelado após CI
O commit `8892980eccfb623de30f3803a7550aa1fa31a9c6` passou pela CI da Fase 3 #41369858499 (21 testes aprovados) antes de ser fixado literalmente em `.github/workflows/phase3-on-label.yml`. `git rev-parse HEAD` verifica o commit, e a guarda do CI exige o mesmo valor literal em checkout e `APPROVED_EVALUATOR_SHA`. O workflow roda somente em `pull_request.labeled` da PR #3 com etiqueta `neural-phase3-generalization-approved`, sem execução por push. CI #41370024631 (24 testes) e teste negativo #41370045092 (skipped) aprovaram a infraestrutura. Continua vedado executar a campanha de 1.050 episódios sem nova autorização do Chat 1.
