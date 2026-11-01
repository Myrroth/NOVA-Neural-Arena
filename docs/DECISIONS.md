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
