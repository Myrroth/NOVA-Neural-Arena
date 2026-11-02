# NOVA Neural Arena — Fase 3: generalização por tamanho de tabuleiro

**Status (2026-11-01): APENAS PREPARAÇÃO TÉCNICA. Campanha de 1.050 episódios NÃO autorizada nem executada.**

## Autoridade e cadeia de proveniência

- Decisão do Chat 1: https://github.com/Myrroth/NOVA-Neural-Arena/pull/2#issuecomment-6566397649.
- Nova branch: `feat/phase3-ood-board-sizes`, baseada em `feat/phase2-frozen-evaluation` (PR #2).
- Código do simulador e DQN intactos, código original de treino `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`.
- Fonte única dos 3 modelos: piloto [#41350959549](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549), tentativa 1, artifacts seed 11 (ID `13101471957`), seed 42 (`13101072100`), seed 73 (`13100988601`). Checkpoints SHA-256 aprovados:
  - 11: `ff132bc23aaf5b56d00679a07961d23c2de30bafa7bbaa4e99ff966b61a49130`
  - 42: `0623256abf3b068f212bacaa5c46f8bbcc6aabf2aeaa0f7ef5ecdc0ec8120cbe`
  - 73: `2899df84afe4512ca127bf3367490a5bc42c88f3ad7868daf8195ed419ecb8fe`
- O carregador reutiliza `arena.evaluation.frozen_phase2.verify_artifact_directory`: sete hashes de `SHA256SUMS`, hash do modelo, origem, configuração e versão conferidos antes de `torch.load(..., weights_only=True)`. Verifica integridade a cada execução, sem editar os ZIPs originais.
- Preservar ZIPs fora do Actions antes de **2026-12-01, ~21:15 UTC**. A avaliação da Fase 2 [#41367188257](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41367188257) expira aproximadamente em 2026-12-02 00:27 UTC.

## Protocolo pré-registrado e controle

- **7 políticas**: 3 QNetwork greedy treinadas (11/42/73); 3 controles QNetwork não treinados com `DQNConfig` e inicializações controladas nas mesmas seeds; 1 política aleatória.
- **3 tamanhos**: 8×8, 10×10 (dentro da distribuição de treino), 12×12.
- **50 seeds ambientais**: 300001 a 300050, compartilhadas por todas as políticas e tamanhos.
- **200 passos máximos**, mesma observação vetorial (11 dimensões), mesmas ações (3 relativas), mesma função de recompensa e mesmo spawn padrão.
- **7 × 3 × 50 = 1.050 episódios previstos**, nenhum nesta fase de preparação. Não alterar regras, posições/direções iniciais ou treinamento. A arquitetura recebe a mesma observação 11D em qualquer tamanho e não requer novos pesos.
- A seed numérica idêntica em tabuleiros de tamanhos diferentes **não implica o mesmo cenário físico**, posições de comida, trajetórias ou sorteio equivalente. Comparações devem ser condicionadas ao tamanho e feitas com cautela.
- A alteração de tamanho muda o espaço de estados e a distribuição de posições possíveis; o teste 8×8/12×12 é uma mudança de ambiente proposta, não demonstra transferência para outros jogos.

## Métricas e definições

Cada linha de `episodes.csv` inclui `policy_id`, `policy_kind`, `training_seed`, `initialization_seed`, `environment_seed`, `board_width`, `board_height`, `max_steps`, `score` (alimentos), `reward`, `steps`, `terminated`, `truncated`, `terminal_reason` (`collision` / `step_limit` / `board_filled`), `first_food_step` (primeiro passo 1-based com crescimento de `score`, vazio sem alimento), `food_per_100_steps = 100*score/steps` (eficiência **por episódio**, não taxa de sobrevivência), `model_sha256`, `artifact_id`.

- `summary.json` tem estatísticas descritivas separadas por tamanho × política, média/mediana/desvio de score, reward, steps e alimentos/100 passos, número de episódios sem alimento, primeira coleta entre partidas em que ocorreu e motivos de término. Evitar estatísticas agregadas que misturem 8/10/12 indevidamente.
- `config.json` registra seeds, dimensões, versões, SHA dos 3 modelos, IDs de artifacts, proveniência e garantia de pesos congelados. `SHA256SUMS` verifica os três arquivos científicos; `evaluation-code-commit.txt` e `original-training-commit.txt` registram os commits na execução futura, e `pip-freeze.txt` registra dependências efetivas.
- A análise inferencial e comparações pareadas por tamanho cabem **somente ao Chat 4**, reconhecendo que existem apenas 3 modelos treinados independentes. Nem `step_limit` nem longevidade sem alimentação significam vitória.

## Garantias de imutabilidade

- `model.eval()`, `model.requires_grad_(False)`, `torch.inference_mode()`, comparação de fingerprints dos parâmetros antes/depois e ausência de gradientes; não criar otimizadores, não salvar checkpoints, não executar `train`.
- `expected_scenarios()` define a matriz sem rodar episódios, rejeita tamanhos/seeds desconhecidos ou duplicados. A CLI **não aceita parâmetros para diminuir/ampliar** as 50 seeds nem as dimensões; o caminho de teste sintético chama apenas subconjuntos internos.
- CLI exige `--authorized`, mas a autorização real é o workflow separado `pull_request.labeled` com a etiqueta exata **`neural-phase3-generalization-approved`** na PR #3. Não aplicar até autorização explícita do Chat 1.
- **Congelamento realizado:** commit auditado do avaliador `8892980eccfb623de30f3803a7550aa1fa31a9c6` aprovado em [CI #41369858499](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41369858499); o workflow `phase3-on-label.yml` faz checkout **literal** desse SHA, verifica `git rev-parse HEAD` e confere Snake/DQN/treinador em relação ao piloto. Alterações posteriores na PR não mudam o código de avaliação executado. Mudanças no próprio workflow exigem nova revisão do Chat 1. Nenhum `push` ou `pull_request_target` autoriza avaliação.
- Permissões mínimas (`contents: read`, `actions: read`), timeout e artifact de retenção limitada; novas alterações no workflow ou código necessitam nova revisão de autorização.

## Validação de infraestrutura vs execução

- `tests/test_phase3_generalization.py`: **checkpoints fabricados sinteticamente, não treinados**, um cenário por tamanho em smoke test (21 episódios sintéticos por repetição), testes de consistência de matriz **sem executar seus 1.050 casos**, indicadores de primeira comida/eficiência, integridade, ausência de optimizer, determinismo e guardas de CLI.
- `.github/workflows/phase3-tests.yml`: executa apenas esses testes + testes do Snake, com Gymnasium e PyTorch oficiais, e baixa os três artifacts **somente para conferir hashes e carregar os pesos sem jogar**. Não executa o avaliador oficial nem campanhas.
- **Concluído:** CI inicial #41369858499 — 21 testes aprovados; workflow fixado em SHA literal. CI após congelamento [#41370024631](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370024631) — **24 testes aprovados e checkpoints oficiais verificados sem avaliação**. Etiqueta neutra `documentation` causou [skipped #41370045092](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370045092), depois foi retirada. Apenas Chat 3, com autorização **posterior** do Chat 1, poderá adicionar a etiqueta de campanha.

**Restrições:** nenhuma execução de 1.050 episódios, nenhum merge, nenhuma mudança nas PRs #1/#2 ou na NOVA original.
