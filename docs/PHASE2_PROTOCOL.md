# Fase 2 — avaliação congelada do Snake (somente proposta)

Estado em 2026-11-01: infraestrutura implementada, **campanha científica NÃO executada**; aguarda aceite e nova etiqueta específica pelo Chat 1.

## Origem e proteção
- Decisão do Chat 1: https://github.com/Myrroth/NOVA-Neural-Arena/pull/1#issuecomment-6564203362.
- Primeiro piloto: Actions run `41350959549`, tentativa 1, commit `24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`; código Snake/DQN/training não alterado.
- Artifacts originais (expiram em torno de **2026-12-01 21:15 UTC**):
  - seed 11: ID `13101471957`, checkpoint SHA256 `ff132bc23aaf5b56d00679a07961d23c2de30bafa7bbaa4e99ff966b61a49130`.
  - seed 42: ID `13101072100`, checkpoint SHA256 `0623256abf3b068f212bacaa5c46f8bbcc6aabf2aeaa0f7ef5ecdc0ec8120cbe`.
  - seed 73: ID `13100988601`, checkpoint SHA256 `2899df84afe4512ca127bf3367490a5bc42c88f3ad7868daf8195ed419ecb8fe`.
- Os arquivos originais são baixados por nome **e run ID fixos**, usando `actions/download-artifact@v8` com `actions:read`. Nenhum arquivo de peso fica commitado no Git.
- Verificação falha ao primeiro erro de `SHA256SUMS`, hash do modelo esperado, `commit.txt`, `provenance.json`, seed, configuração, identidade de run/attempt e dependências. Só depois ocorre `torch.load(weights_only=True)`.

## Desenho fixo
- Snake **10×10** com `max_steps=200`, exclusivamente `ENVIRONMENT_SEEDS=200001..200050`.
- **7 políticas**, com condições pareadas por seed ambiental:
  - 3 DQN treinadas: checkpoints 11, 42 e 73 (online weights originais, greedy e eval).
  - 3 controles neurais não treinados: mesma classe `QNetwork` e mesmos `DQNConfig` dos modelos, inicializações reproduzíveis com seeds **11, 42 e 73**. Iniciar a rede sem experiência nem step de otimizador.
  - 1 política aleatória reprodutível: `random.Random(90000000 + environmental_seed).randrange(3)`, compartilhada para as comparações.
- Cada episódio registra `score`, `reward`, `steps`, `terminated`, `truncated`, `terminal_reason` (`collision`, `step_limit`, `board_filled`), tipo/id de política, SHA do checkpoint, seed de treinamento, seed da inicialização e ID do artifact.
- O runner usa `torch.inference_mode()`, `network.eval()`, `requires_grad_(False)`, e rejeita qualquer modificação de pesos ou criação de gradientes; não chama `DQNAgent.learn` ou `optimizer.step`.
- **Total da campanha prevista**: 7 × 50 = **350 episódios**, sendo 150 de modelos treinados, 150 de controles não treinados e 50 de random. Não foram executados nesta tarefa.
- Arquivos de avaliação previstos: `episodes.csv`, `summary.json`, `config.json`, `SHA256SUMS`, `evaluation-code-commit.txt`, `original-training-commit.txt`, `pip-freeze.txt`. Sem mesclar com os artifacts originais. `summary.json` é resumo descritivo, análise inferencial cabe ao Chat 4.
- Limites: mesmos cenários ambientais não implicam ações/trajetórias iguais. Não há teste OOD, transferência entre jogos ou nova aprendizagem. Somente **3 treinos independentes**.

## Tests vs. evaluation
- `tests/test_phase2_frozen.py` fabrica pequenos checkpoints **sintéticos e não treinados** para testar hashes, integridade, provenance, carregamento com `weights_only=True`, modo eval, dois ambientes seguros, mesma seed, ausência de optimizer e registro de desfechos.
- `.github/workflows/phase2-tests.yml` roda apenas `pytest tests/test_phase2_frozen.py tests/test_snake.py`, automaticamente em PR direcionada à branch da PR #1 (exclui testes legados que fazem microtreinamento). **Baixa os três artifacts oficiais somente para conferir os sete hashes e carregar pesos congelados**, sem fazer inferência ou episódios científicos. Essa checagem oficial passou no run [#41353674728](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353674728) antes do ajuste que restringe a seleção de testes.
- `.github/workflows/phase2-on-label.yml` tem **somente** `pull_request: labeled`, nunca `push` ou `pull_request_target`, e o job exige exatamente a etiqueta **`neural-phase2-evaluation-approved`** na nova PR #2 (mesmo repositório e branches corretos). Checkout fixo no commit previamente validado `f9e1b171a8c045f8bd7c874432114e4bc654665e` (CI #41353905275), independente de pushes futuros da PR; `git rev-parse HEAD` verifica o SHA literal, e o runner valida binários de Snake e DQN contra o commit do piloto e executa a avaliação científica apenas quando Chat 1 autorizar a aplicação da etiqueta.
- Uma etiqueta neutra deve produzir `skipped`. **Não aplicar a etiqueta autorizadora neste Chat 2.**
- Limite do workflow: 35 min, permissões `contents: read` e `actions: read`, output retention 30 dias. Checkpoints originais podem expirar; preservá-los externamente antes de 2026-12-01.

## Para Chat 1 — decisão pendente
1. Revisar novos arquivos, CI de testes e protocolo antes de autorização.
2. Após aprovação explícita, Chat 3 poderá aplicar `neural-phase2-evaluation-approved` **uma vez** na PR #2 e acompanhar a execução; Chat 4 fará análise científica.
3. Não fazer merge em nenhuma PR antes de autorização; `main` e IA NOVA original permanecem intactas.

## Congelamento do avaliador para aprovação da campanha
- `phase2-on-label.yml` usa `ref: f9e1b171a8c045f8bd7c874432114e4bc654665e` e `EXPECTED_HEAD: f9e1b171a8c045f8bd7c874432114e4bc654665e` (não o `github.event.pull_request.head.sha` variável). O arquivo do workflow pode receber commits de documentação sem mudar o código avaliado, mas **qualquer mudança do próprio workflow exige nova auditoria antes da etiqueta**.
- O avaliador executado será aquele do commit **f9e1b171a8c045f8bd7c874432114e4bc654665e**, com teste oficial aprovado [#41353905275](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41353905275). Após checkout o runner confirma `git rev-parse HEAD`; se divergir, falha antes de carregar os modelos.
- A CI da PR verifica o SHA literal em duas posições e mantém os controles opt-in. Não aplicar `neural-phase2-evaluation-approved` até autorização posterior do Chat 1.
