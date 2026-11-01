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
