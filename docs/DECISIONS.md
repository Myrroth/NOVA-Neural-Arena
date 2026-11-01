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
