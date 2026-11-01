# NOVA Neural Arena — Contexto oficial

**Versão:** 0.1 — laboratório independente, iniciado em 2026-11-01.

## Objetivo
Investigar se um agente neural aprende a observar, agir, receber feedback e adaptar-se a novos jogos por tentativa e erro. O objetivo de longo prazo é investigar transferência entre ambientes novos, sem prometer solução universal.

## Princípios
1. Interface consistente de agente/ambiente; ambientes podem ter adaptadores próprios.
2. Aprender por interação, não por respostas programadas especificamente para resolver Snake.
3. Experimentos locais, seguros e sem acesso a contas externas ou dados pessoais.
4. Registrar seeds, versões, hiperparâmetros, métricas e checkpoints.
5. Avaliar em seeds e configurações não vistos e comparar com políticas aleatórias.
6. Separar medições de hipóteses: um sucesso em Snake não prova IA geral.
7. Evitar trabalho longo, gastos e runs automáticos sem autorização.
8. Manter resultados importantes fora de artifacts temporários do Actions.

## Roadmap
- Fase 0: infraestrutura Python, padrões Gymnasium, testes, seeds, métricas e CI manual.
- Fase 1: Snake vetorial e DQN, comparação com baseline aleatória.
- Fase 2: generalização dentro do Snake em tabuleiros/configurações inéditos.
- Fase 3: Pong com interface de ambiente compartilhada.
- Fase 4: simulador Agar-like local, sem usar servidores externos.
- Fase 5: comparar políticas individuais e multitarefa, transferência e esquecimento.
- Fase 6: modelo de mundo para prever próximos estados/recompensas.
- Fase 7: possível integração com IA NOVA, somente mediante autorização explícita.

## Estrutura
- arena/envs: simuladores Gymnasium;
- arena/agents: políticas e DQN;
- arena/training: treino e checkpoints;
- arena/evaluation: validação em sementes não vistas;
- arena/shared: componentes comuns futuros;
- tests: testes automatizados;
- experiments: métricas/checkpoints locais (ignorados pelo Git);
- docs: fonte de verdade;
- .github/workflows: testes e treino curto manuais.

## Métricas e análise
Snake: pontuação média/mediana/desvio, sobrevivência e recompensas. Treino e avaliação usam seeds distintas, com avaliação emparelhada nas mesmas seeds para DQN e baseline. Pesquisar robustez de tamanhos de tabuleiro, riscos de esquecimento e múltiplas rodadas de treino.

## GitHub e segurança
Desenvolver em branches e PRs, sem merges automáticos. Workflows inicialmente somente workflow_dispatch. Ações padrão CPU e sem acesso a segredos. Não alterar NOVA, não conectar a jogos externos, não publicar dados privados. Artifacts podem expirar.

## Continuidade
Antes de qualquer desenvolvimento, ler PROJECT_CONTEXT.md, STATUS.md, DECISIONS.md e PRs relevantes. Atualizar STATUS com resultados reais, pendências e PR/commit; DECISIONS com novas escolhas técnicas. Conversas do ChatGPT são contexto auxiliar, não fonte de verdade.

## Limites
Pesquisa online (RAG) não equivale a treinar pesos neurais. Modelo de mundo estima consequências sob hipóteses do ambiente, não prevê eventos reais com certeza. Nenhuma promessa de generalização até experimentos adequados.
