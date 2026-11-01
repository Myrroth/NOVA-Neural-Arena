# STATUS — NOVA Neural Arena

Atualizado em: 2026-11-01

- PR: https://github.com/Myrroth/NOVA-Neural-Arena/pull/1 (rascunho, sem merge).
- Commit inicial: https://github.com/Myrroth/NOVA-Neural-Arena/commit/aaab24c46d3663ba2e4728c25f8ca4e2947a0fee

## Implementado na branch feat/snake-dqn-laboratory
- Estrutura Python independente, sem tocar a IA NOVA.
- Snake Gymnasium com seeds, render ANSI, tabuleiros configuráveis, observação parcial 11D e ações relativas.
- DQN PyTorch com replay, target network, epsilon-greedy, checkpoint model.pt.
- Treino CLI e arquivos episodes.csv, evaluation.json, run.json, com versões, seeds e hiperparâmetros.
- Baseline aleatória e avaliação pareada com seeds de treino/avaliação separadas.
- Dez testes de ambiente, treinamento, checkpoint e benchmark, mais um teste de contrato com o checker oficial Gymnasium.
- Workflows manuais em .github/workflows/ci.yml e train.yml; ainda não executados no GitHub.

## Validação executada localmente
- 10 testes aprovados, 1 teste do checker oficial ignorado por dependência ausente, em Python com NumPy 2.3.5, PyTorch 2.10.0 CPU e pytest 9.0.2.
- **Limitação:** Gymnasium ausente no runtime de validação; utilizado stub local de compatibilidade (fora do repositório). Os testes **não** validam integração com Gymnasium oficial.
- Treino demonstrativo com 100 episódios, seed 42, 10x10, até 200 passos, 30 avaliações nas seeds 100042..100071: 2.278 updates; pontuação média DQN greedy 18,30 vs aleatória 0,2333, mediana 19 vs 0. Desvio de pontuação 4,489 vs 0,568. Resultados locais provisórios; necessitam reprodução com dependências oficiais antes de conclusão.
- Repetido o treinamento completo com mesma seed e parâmetros: `episodes.csv`, `evaluation.json` e `run.json` ficaram byte-a-byte idênticos. Checkpoint gerado localmente; ainda não publicado como artifact de Actions.

## Problemas e limites conhecidos
- Actions ainda não disparados; status operacional CI pendente de confirmação. **Limitação do GitHub:** workflows `workflow_dispatch` precisam existir na branch padrão para disparo normal pela interface; como esta implementação está somente na branch da PR, não é possível tratá-los como operacionais antes da aprovação/integração do workflow. Evitar alterar `main` fora de PR.
- Teste com Gymnasium real e env_checker pendente.
- Apenas Snake, observação parcial, sem transferência ou avaliação de generalização.
- Checkpoint não inclui replay buffer ou RNG para retomada exata.
- Treino curto e uma seed de treinamento não bastam para conclusões gerais.

## Próximos passos
1. Disparar Manual tests e Manual short DQN train na branch; conferir logs/artifacts.
2. Validar o ambiente com Gymnasium env_checker e mais testes de borda.
3. Repetir treinos com múltiplas seeds e configurações Snake não vistas.
4. Melhorar política e metodologia antes de Pong e comparação entre jogos.
5. Nunca fazer merge na main sem autorização.
