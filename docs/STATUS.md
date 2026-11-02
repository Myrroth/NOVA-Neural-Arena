# STATUS — NOVA Neural Arena

Atualizado em 2026-11-01 — Chat 2 / infraestrutura de piloto em PR #1 (sem merge).

## Código de referência e decisão

- Repositório: `Myrroth/NOVA-Neural-Arena`; PR: https://github.com/Myrroth/NOVA-Neural-Arena/pull/1; branch `feat/snake-dqn-laboratory`.
- **Commit imutável autorizado para experimentar:** `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`. O workflow novo está em commit posterior, mas **faz checkout desse SHA**; Snake, DQN, benchmark e treino não são alterados por este trabalho.
- Decisão Chat 1: comentário https://github.com/Myrroth/NOVA-Neural-Arena/pull/1#issuecomment-6563787807.
- Código de referência validado com **13 testes + env_checker oficial** (Gymnasium 1.4.0): https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41330854039 (sucesso; 2 warnings não bloqueantes de `env.spec`).
- Chat 3 relatou bloqueio local de DNS na instalação oficial; **nenhum** dos pilotos 11/42/73 foi executado neste momento, sem métricas e sem checkpoints experimentais. Chat 4 não pode analisar eficácia até receber dados reais.

## Infraestrutura adicionada na PR #1

Workflow: `.github/workflows/neural-pilot-on-label.yml`.

- Exclusivamente evento `pull_request: types: [labeled]`, base `main`; **não** há `push`, `pull_request_target` ou acionamento manual implícito neste workflow.
- O job só roda se a etiqueta recém-adicionada for **`neural-pilot-approved`**, PR número **1**, origem/destino do mesmo repositório e head `feat/snake-dqn-laboratory`.
- No runner Ubuntu 24.04 com Python 3.11, realiza checkout por SHA fixo `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463` e confere `git rev-parse HEAD`, sem persistir credenciais.
- Instala `torch==2.10.0` do índice **oficial de wheels CPU PyTorch**; instala dependências usando `requirements-ci.txt` (Gymnasium 1.4.0, NumPy 2.3.5, pytest 9.0.2); executa `pip check`, `env_checker` real e `pytest` no código aprovado antes do treino.
- Três jobs independentes por matriz (`fail-fast: false`, máximo 3 em paralelo), seeds **11, 42 e 73**. Cada job treina **80 episódios**, avalia **20 partidas**, em Snake **10×10** até **200 passos**.
- Diretórios `experiments/chat3/seed_11/`, `seed_42/`, `seed_73/`. Cada artifact (retenção **30 dias**) contém `model.pt`, `episodes.csv`, `evaluation.json`, `run.json`, `pip-freeze.txt`, `commit.txt`, `provenance.json` e `SHA256SUMS`, quando as etapas terminam com sucesso.
- `provenance.json` inclui SHA fixo, seed, protocolo, identificadores do run e PR; o job valida arquivos, quantidades de treino e avaliação, avaliação em seeds distintas, além do commit.
- `permissions: contents: read`, timeout 35 min/job, concurrency por PR sem cancelar job em curso. Não publica código, não usa segredos.

## Verificações e resultado de infraestrutura

- **Confirmado anteriormente:** CI do código em `24bd38...` aprovado em GitHub Actions com Gymnasium e PyTorch oficiais; log e artifact de versões disponíveis. CI após adicionar workflow e docs https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41349905467 também **aprovada**.
- **Configuração adicionada:** workflow separado foi commitado somente na branch da PR. Não foram feitos merges ou mudanças em `main`; nenhum treinamento foi iniciado pelo Chat 2.
- **Gatilho por etiqueta testado realmente:** etiqueta temporária `documentation` aplicada à PR #1 e depois removida. O GitHub criou https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41349938598 (`pull_request`), **conclusão `skipped`**: o job condicionado a `neural-pilot-approved` não executou passos nem treinamento. Isso valida o evento e o bloqueio para etiqueta não autorizadora, **não comprova a execução do caso positivo**.
- **Estado do piloto:** **gatilho por etiqueta comprovado em teste negativo; infraestrutura liberada ao Chat 3 para executar o caso autorizado**. Nenhum treino foi disparado pelo Chat 2. A primeira execução positiva validará instalações/matriz/checkpoints/artifacts reais.
- Observação: workflows executados via `pull_request` usam ref de merge por padrão; nosso step `checkout` sobrescreve o ref com SHA fixo para evitar executar o merge. `pull_request_target` foi deliberadamente rejeitado.

## Como o Chat 3 iniciará (após liberação)

1. Confirmar que o workflow `Approved Snake pilot (PR label)` consta na PR #1 e que o CI padrão segue aprovado.
2. Se a etiqueta `neural-pilot-approved` ainda não existir no repositório, criá-la pela interface GitHub (`Issues > Labels > New label`) **sem aplicá-la** antes de decidir iniciar.
3. Na PR #1, usar a seção **Labels** para **adicionar** `neural-pilot-approved`. **Esse é o ato que dispara os três jobs e consome minutos de Actions.** Não aplicá-la por automação usando `GITHUB_TOKEN` (eventos originados desse token normalmente não disparam novos workflows).
4. Acompanhar a execução na aba **Actions** e conferir três jobs, seeds distintas, checkout SHA fixo, testes oficiais, avaliação e artifact por seed.
5. Baixar os três artifacts antes de 30 dias e encaminhar `model.pt`, `episodes.csv`, `evaluation.json`, `run.json`, `pip-freeze.txt`, `provenance.json`, `commit.txt`, `SHA256SUMS` ao Chat 4. Caso seja necessário repetir, remover a etiqueta e aplicá-la de novo **somente com nova autorização**.

## Pendências e limites

- Confirmado disparo de `pull_request.labeled` com etiqueta neutra. **Ainda pendente:** execução positiva somente pelo Chat 3, com `neural-pilot-approved`, e confirmação de três artifacts completos. Se o caso positivo falhar, diagnosticar logs antes de repetir; se o gatilho não funcionar, consultar o Chat 1 sobre alternativa de runner confiável sem alterar `main`.
- O antigo workflow `train.yml` utiliza `workflow_dispatch` e segue indisponível pela interface sem existir na branch padrão; o novo é opção alternativa opt-in.
- Dependências **diretas** pinadas; `pip-freeze.txt` registra transitivas efetivamente instaladas. O checkout da mesma versão não garante reprodutibilidade bit a bit entre máquinas.
- Checkpoint de treinamento aprovado não salva replay/RNG para retomada exata. Desfechos `terminated`/`truncated` ausentes no arquivo de avaliação são uma melhoria futura, não alterar durante o piloto fixo.
- Execução e avaliação científica reservadas ao Chat 3 e Chat 4. Sem campanhas extensas; sem merge; NOVA original intocada.


## 2026-11-01 — Chat 2: preparação isolada Fase 2 (nova PR em rascunho)
- Decisão do Chat 1: https://github.com/Myrroth/NOVA-Neural-Arena/pull/1#issuecomment-6564203362.
- Piloto nº 001 concluído (run 41350959549): três jobs oficiais passaram; artefatos e seus hashes auditados no Relatório Científico nº 002.
- Nova branch `feat/phase2-frozen-evaluation`, originada de `feat/snake-dqn-laboratory`; **Snake/DQN/treinador e resultados anteriores preservados**.
- Infraestrutura proposta: `arena/evaluation/frozen_phase2.py` (verifica hashes e proveniência dos três checkpoints, avalia em modo congelado e registra causas explícitas dos desfechos); `tests/test_phase2_frozen.py` (somente checkpoints sintéticos, subset de 2 seeds em smoke tests); `phase2-tests.yml` (CI de testes) e `phase2-on-label.yml` (avaliação condicionada à nova etiqueta, NUNCA aplicada nesta etapa).
- Protocolos e hashes documentados em `docs/PHASE2_PROTOCOL.md`.
- **Campanha dos 50 ambientes NÃO executada; checkpoints oficiais NÃO modificados; nenhum novo treinamento.** Testes de integração com arquivos oficiais pendem da execução futura autorizada pelo Chat 1.
- Artifact IDs: 13101471957, 13101072100, 13100988601; expiração por volta de **2026-12-01 21:15 UTC**. Preserve ZIPs oficiais antes disso.
- **Estado:** código submetido para CI de infraestrutura; **aprovação científica e autorização de avaliação pendentes**. Nenhum merge na PR #1 ou nova PR, sem alteração da NOVA original.

### Validação comprovada da infraestrutura da Fase 2 (2026-11-01)
- [CI inicial #41353536523](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353536523): **22 testes aprovados, 2 avisos Gymnasium não bloqueantes** (incluía ainda testes legados de treino curto da PR #1; não executou a campanha de Fase 2 nem treinou os checkpoints oficiais).
- [CI com artefatos reais #41353674728](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353674728): **22 testes aprovados**; baixou os 3 artifacts oficiais do piloto, conferiu `SHA256SUMS`, modelo esperado/proveniência e **carregou os 3 checkpoints sem executar episódios/inferência**. Seed 11/42/73 PASS nos logs.
- [Teste do gatilho com `documentation` #41353590209](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353590209): run **skipped**; nenhuma avaliação científica disparada. A etiqueta foi removida.
- **Endurecimento concluído:** workflows da Fase 2 agora executam apenas `tests/test_phase2_frozen.py` e `tests/test_snake.py`, excluindo testes legados que invocavam microtreinamentos. [CI #41353805183](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353805183): **18 testes aprovados, 2 avisos não bloqueantes; as três verificações de integridade dos artifacts e carregamento seguro das redes também passaram.** A campanha de 350 episódios nas 50 seeds comuns permanece **NÃO AUTORIZADA/NÃO EXECUTADA**.

### 2026-11-01 — Correção de controle de versão obrigatória (Chat 2)
- Decisão do Chat 1: https://github.com/Myrroth/NOVA-Neural-Arena/pull/2#issuecomment-6565513811. Protocolo científico aprovado **sem autorização de executar a campanha**.
- Código avaliador efetivamente auditado e aprovado na CI https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353905275: **`f9e1b171a8c045f8bd7c874432114e4bc654665e`** (18 testes; 3 checkpoints oficiais conferidos/carregados, nenhuma campanha).
- `.github/workflows/phase2-on-label.yml`: `actions/checkout@v7` usa agora `ref: f9e1b171a8c045f8bd7c874432114e4bc654665e` **literal**, e `EXPECTED_HEAD: f9e1b171a8c045f8bd7c874432114e4bc654665e` para `git rev-parse HEAD`. Não utiliza mais o SHA mutável do HEAD da PR.
- Teste de regressão estático `tests/test_phase2_workflow_pin.py` incluído na CI da PR para exigir ambas as constantes, o gate por etiqueta específica e ausência de push, `pull_request_target` ou treinamento.
- Garantia de código: uploads/documentação futuros da PR não mudam o commit executável; alterações no próprio workflow demandam nova revisão do Chat 1 antes da aplicação da etiqueta. O SHA do workflow no evento `labeled` não é necessariamente o SHA de checkout (este é congelado).
- Nenhuma etiqueta de autorização foi aplicada. `main` e PR #1 preservadas, sem merge e sem tocar na NOVA original. A campanha de **350 episódios permanece bloqueada até autorização do Chat 1**.


## 2026-11-01 — Fase 3: preparação da avaliação entre tamanhos (Chat 2)
- Diretriz Chat 1 (última decisão da PR #2): https://github.com/Myrroth/NOVA-Neural-Arena/pull/2#issuecomment-6566397649; Fase 2 encerrada cientificamente, **Fase 3 sem autorização de executar**.
- Nova branch `feat/phase3-ood-board-sizes`, derivada da PR #2, para PR #3 em rascunho; **PR #1/#2, main, modelos e Snake/DQN/treinador originais intactos**.
- Implementado `arena/evaluation/phase3_generalization.py`: três modelos oficiais congelados + três QNetwork não treinadas + aleatória; sizes (8,10,12), seeds ambientais 300001..300050, máximo 200 passos. Matrix prevista 1.050 episódios, **não executados**.
- Métricas: score/reward/steps, desfecho, primeiro alimento, alimentos por 100 passos, proveniência e estatísticas separadas por dimensão e política.
- Testes sintéticos `tests/test_phase3_generalization.py` e CI isolada `phase3-tests.yml` com dependências oficiais e audit do download/checkpoints **sem episódios com modelos oficiais**.
- Protocolo e limites em `docs/PHASE3_PROTOCOL.md`. A CI da PR verificará compatibilidade; depois de aprovada será congelado o commit do avaliador no workflow opt-in da Fase 3.
- Checkpoints oficiais ID 13101471957/13101072100/13100988601 com retenção prevista até 2026-12-01 21:15 UTC. Preservar ZIPs originais fora da retenção temporária.


### 2026-11-01 — Auditoria final de infraestrutura Fase 3 (sem campanha)
- Commit do avaliador **congelado após CI**: `8892980eccfb623de30f3803a7550aa1fa31a9c6`, [run #41369858499](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41369858499), **21 testes aprovados**, 2 avisos do Gymnasium não bloqueantes. Hash/proveniência e carregamento congelado dos 3 checkpoints oficiais 11/42/73 conferidos sem partidas oficiais.
- `.github/workflows/phase3-on-label.yml` usa checkout **literal** `ref: 8892980eccfb623de30f3803a7550aa1fa31a9c6` e `APPROVED_EVALUATOR_SHA: 8892980eccfb623de30f3803a7550aa1fa31a9c6` para `git rev-parse HEAD`, além de comparação dos blobs Snake/DQN/trainer contra o piloto. A etiqueta exclusiva `neural-phase3-generalization-approved` exige PR #3, branch e repositório corretos. Não há gatilhos `push` nem `pull_request_target`.
- `tests/test_phase3_workflow_pin.py` protege o SHA, a guarda do checkout, etiqueta exata, limites e ausência de treinamento; incluído em `phase3-tests.yml`.
- [CI após congelamento #41370024631](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370024631): **24 testes aprovados, 2 avisos não bloqueantes**; os 3 arquivos oficiais baixados, SHA256/proveniência conferidos e pesos carregados congelados. Zero episódios de avaliação oficial.
- [Gatilho negativo #41370045092](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370045092): etiqueta neutra `documentation` → job da Fase 3 **skipped**, nenhum jogo; etiqueta removida.
- **Pendência:** Chat 1 revisar ciência e infraestrutura, então autorizar (ou não) execução **única** por Chat 3. Nenhuma etiqueta autorizadora aplicada, nenhum merge ou alteração dos modelos originais.


## 2026-11-01 — Chat 2: preparação isolada da Fase 4 (robustez de estado inicial)
- Decisão do Chat 1: https://github.com/Myrroth/NOVA-Neural-Arena/pull/3#issuecomment-6566860199. Fase 3 encerrada; **somente implementação da Fase 4 autorizada, nenhuma avaliação oficial**.
- Nova branch `feat/phase4-startstate-robustness`, derivada da PR #3; preparar PR #4 em rascunho.
- Ambiente adaptado `StartStateSnakeEnv`, em `arena/evaluation/phase4_startstates.py`, cria corpo de 3 segmentos atrás da cabeça para 3 posições e 4 direções, sem tocar `arena/envs/snake.py`. Comida sorteada **depois** de posicionar corpo, sem sobreposição, e posição real registrada.
- Quadro 10×10; seeds inéditas 400001..400050; sete políticas congeladas 11/42/73 e controles; limite 200 passos; matriz de 4.200 episódios é **apenas plano não executado**.
- `expected_scenarios()` calcula 4.200 chaves sem executar; testes com modelo sintético verificam 12 condições, determinismo, paridade com o Snake padrão, integridade, primeira comida, eficiência, ausência de treino e pequenas simulações.
- CI `phase4-tests.yml` instala Gymnasium/PyTorch oficiais, valida apenas testes sintéticos e confere 3 checkpoints oficiais sem episódios. Próxima etapa: primeira CI verde, então SHA imutável do avaliador em workflow opt-in exclusivo da Fase 4.
- Detalhes em `docs/PHASE4_PROTOCOL.md`; nenhum merge e nenhuma mudança na NOVA original, modelos, treinadores ou PRs anteriores.


### 2026-11-01 — Fase 4: avaliador congelado e infraestrutura oficialmente validada
- Commit avaliado e congelado: `8b4b6696e980191349e803218af6e6d55d8ff7e9`. [CI #41372851622](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41372851622): **42 testes aprovados, 2 avisos Gymnasium informativos**, 3 checkpoints oficiais 11/42/73 íntegros e carregados congelados sem episódios.
- Workflow `.github/workflows/phase4-on-label.yml` usa literal `ref: 8b4b6696e980191349e803218af6e6d55d8ff7e9` e `APPROVED_EVALUATOR_SHA: 8b4b6696e980191349e803218af6e6d55d8ff7e9`, checagem `git rev-parse HEAD` e comparação de Snake/DQN/treinador contra o piloto original. **Não segue o HEAD mutável da PR**, nem usa `push` ou `pull_request_target`.
- [CI endurecida #41373055405](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41373055405): **45 testes aprovados, 2 avisos não bloqueantes**, incluindo `tests/test_phase4_workflow_pin.py` e auditoria dos 3 artifacts (sem jogar com eles).
- [Teste negativo #41373064866](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41373064866): etiqueta `documentation` gerou `skipped`, removida após o teste. Etiqueta autorizadora `neural-phase4-startstate-approved` permanece **NÃO aplicada**.
- Protocolo técnico em `docs/PHASE4_PROTOCOL.md`. Campanha de **4.200 episódios NÃO executada** e **NÃO autorizada**; aguarda revisão do Chat 1. Sem merge, novo treinamento ou modificação da NOVA original.
