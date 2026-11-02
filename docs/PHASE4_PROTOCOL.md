# NOVA Neural Arena — Fase 4: robustez da posição e direção iniciais

**Estado: preparação de infraestrutura SOMENTE. A campanha oficial de 4.200 episódios NÃO está autorizada.**

## Decisão e cadeia de confiança

- Chat 1: https://github.com/Myrroth/NOVA-Neural-Arena/pull/3#issuecomment-6566860199.
- PR #4 (nova, rascunho), baseada na branch da PR #3 `feat/phase3-ood-board-sizes`; conservar PRs #1–#3, `main` e NOVA original sem alterações.
- Manter exatamente os checkpoints oficiais do piloto [#41350959549](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549), commit original `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`.
  - Seed 11, artifact `13101471957`, SHA-256 `ff132bc23aaf5b56d00679a07961d23c2de30bafa7bbaa4e99ff966b61a49130`.
  - Seed 42, artifact `13101072100`, SHA-256 `0623256abf3b068f212bacaa5c46f8bbcc6aabf2aeaa0f7ef5ecdc0ec8120cbe`.
  - Seed 73, artifact `13100988601`, SHA-256 `2899df84afe4512ca127bf3367490a5bc42c88f3ad7868daf8195ed419ecb8fe`.
- `verify_artifact_directory` da Fase 2 valida integralmente `SHA256SUMS` (7 arquivos), SHA oficial de `model.pt`, seed, código original, proveniência, run e versões ANTES de `torch.load(..., weights_only=True)`. `load_verified_model` carrega somente pesos oficiais e valida arquitetura. Sem binários versionados na PR.
- Artefatos de 30 dias exigem preservação externa autorizada: piloto expira ~**2026-12-01 21:15 UTC**, Fase 2 ~**2026-12-02 00:27 UTC**, Fase 3 ~**2026-12-02 01:06 UTC**.

## Hipótese e matriz pré-registradas

As redes treinadas com cabeça (5,5), orientação leste, Snake 10×10 e 200 passos mantêm coleta de comida diante de outras orientações e posições iniciais?

- **Políticas 7:** treinadas 11, 42, 73; não treinadas (mesma QNetwork, mesmo `DQNConfig`, inicializações reproduzíveis 11, 42, 73); uma aleatória `random.Random(90_000_000 + environmental_seed)`.
- **Tabuleiro fixo 10×10**, 200 passos, observação 11D, 3 ações relativas, recompensas e colisões do `SnakeEnv.step` ORIGINAL.
- **Posições da cabeça:** (5,5) — padrão, (3,3), (6,6). **Orientações:** east, south, west, north, na ordem do `DIRECTIONS` original. Total **12 combinações balanceadas**.
- **Sementes ambientais:** 400001..400050 (50), exclusivas da fase. A mesma seed é compartilhada por **todas as 7 políticas dentro de cada condição de início**.
- **Total previsto:** 7 políticas × 12 condições × 50 ambientes = **4.200 episódios EXCLUSIVAMENTE DE AVALIAÇÃO**, sem autorização para executá-los ainda. `expected_scenarios()` calcula o desenho sem executar qualquer episódio e rejeita duplicações/valores fora do protocolo.
- Caso de referência **(5,5, east)**: a adaptação deve produzir corpo, observação e primeira comida **idênticos ao Snake original para a mesma seed** (verificado por teste).
- **A mesma seed numérica em condições de início diferentes não implica a mesma coordenada de comida ou mesmo cenário físico.** A posição de comida efetivamente sorteada é registrada para identificar esse potencial confundimento. Não considerar cenários fisicamente emparelhados entre condições.

## Adaptação isolada

`arena/evaluation/phase4_startstates.py` contém `StartStateSnakeEnv(SnakeEnv)` — uma extensão **somente para avaliação**; `arena/envs/snake.py` permanece idêntico ao commit original. Seu `reset` chama o método oficial de semeadura `gym.Env.reset` (sem realizar o spawn antecipado da classe-base), reconstrói o corpo com **3 posições** `head - n * facing_delta`, `n = 0, 1, 2`, verifica unicidade/limites e **depois** sorteia comida via `_spawn_food()`; confirma que ela está livre e dentro de 10×10. Rejeita `options` e posicionamento arbitrário fora das 12 condições. Herda `step`, `_observation`, `_collision` e recompensa do Snake original, sem reimplementar regras.

Os modelos são congelados com `model.eval()`, `requires_grad_(False)`, `torch.inference_mode()`, fingerprints antes/depois e rejeição de gradientes; **nenhum optimizer, save de modelo ou treinamento**.

## Desfechos, endpoints e limites científicos

- **Endpoint primário:** `score` (número de alimentos) registrado por política × posição × direção × ambiente; comparações futuras de cada condição com o padrão nas **novas seeds da mesma fase**.
- Secundários: `reward`, `steps`, `terminated`, `truncated`, `terminal_reason` (collision/step_limit/board_filled), `food_per_100_steps=100*score/steps`, `first_food_step` (índice 1-based; nulo sem alimento), alimento inicial (`initial_food_x`, `initial_food_y`), head_x/head_y e direção, SHA/ID do checkpoint. `step_limit` não é vitória nem competência.
- `summary.json` discrimina **12 condições × 7 políticas**, sem misturar contextos. Estatísticas descritivas: média, mediana, desvio padrão, extremos, episódios com/sem alimento, passo até primeiro alimento condicional e motivos de término. `config.json` registra matriz, hashes oficiais, versões, seeds, política e proveniência.
- Saída futura: `episodes.csv`, `summary.json`, `config.json`, `SHA256SUMS`, `evaluation-code-commit.txt`, `original-training-commit.txt`, `execution.json`, `pip-freeze.txt`. Hashes SHA-256 cobrem resultados.
- **Chat 4**, após execução autorizada: contrastes treinada vs não treinada/aleatória dentro da mesma condição; posição, direção, interação e diferenças relativas ao padrão com ICs e correção de multiplicidade. Somente **3 modelos treinados independentes**; 4.200 episódios não são 4.200 treinos. Não assumir equivalência, causalidade universal ou transferência para outros jogos.

## Segurança, validação, aprovação

- `tests/test_phase4_startstates.py` cria apenas **checkpoints sintéticos**, confere os 12 spawns, comida livre e determinismo, paridade com Snake padrão, desfechos, métricas, integridade e invariância dos pesos; faz smoke test **2 condições × 1 seed × 7 políticas = 14 episódios sintéticos por repetição**, nunca com os checkpoints oficiais.
- `.github/workflows/phase4-tests.yml` roda exclusivamente testes pequenos, baixa os 3 arquivos oficiais apenas para verificação de SHA/proveniência e carregamento **sem executar partidas com eles**. CI usa Python 3.11, Gymnasium 1.4.0 e PyTorch 2.10.0 CPU oficiais.
- Workflow científico futuro `.github/workflows/phase4-on-label.yml`: gatilho `pull_request.labeled` **somente na PR #4**, etiqueta `neural-phase4-startstate-approved`, sem `push`, `workflow_dispatch` ou `pull_request_target`, permissões somente leitura, timeout, limites de recursos e artifacts de retenção limitada.
- **Avaliador será fixado em SHA literal somente DEPOIS da aprovação da primeira CI**, com `git rev-parse HEAD` e comparação de Snake, DQN, treinador ao original; testes de pin verificarão isso. Mudanças futuras no workflow deverão ser reavaliadas pelo Chat 1.
- **O Chat 1 não autorizou aplicar a etiqueta nem executar os 4.200 episódios**. Chat 3 só executará uma vez após nova decisão explícita; Chat 4 interpreta resultados oficiais. Nenhum merge, novos treinos ou alteração da NOVA original.
