# NOVA Neural Arena — resultado da campanha única 002

A infraestrutura comum foi implementada e treinada nos cinco simuladores preparados. A visão integral de aprender todos os jogos, transferir habilidades e conservar o conhecimento **não foi alcançada** nesta campanha. A execução finita terminou; resultados negativos são preservados, sem novo treinamento automático.

| Jogo | O que os dados sustentam | Critério completo |
| --- | --- | --- |
| Snake | 2,82 alimentos/episódio contra zero nos dois controles; três réplicas positivas e cobertura de início em interior, parede e canto. | PASS |
| Pong | Melhor interceptação e +0,85 ponto líquido contra aleatório. O ganho contra rede não treinada tem intervalo que cruza zero; saldo final ainda −2,15. | FAIL, aprendizado parcial |
| Agar-like | Ganhos pequenos em alimentos/presas, abaixo das margens. Não demonstrou redução de morte por células maiores. | FAIL |
| Flappy | 0,73 cano/episódio contra zero nos controles; melhora nos dois tamanhos de abertura. A margem fixada era +1 cano. | FAIL, aprendizado parcial |
| Corrida | Compartilhado adaptado: 32,38% da volta; mesma arquitetura treinada do zero: 88,37%. Transferência final −55,99 pontos percentuais, IC95% [−86,63; −29,90]. | FAIL |

O controle de corrida treinado do zero completou 75% das voltas avaliadas; o compartilhado, 8,33%. Portanto existe aprendizado no alvo pelo controle, mas não transferência positiva do agente compartilhado. A vantagem relativa de área sob a curva foi −43,03%, e a retenção mínima de 80% falhou nos quatro jogos-fonte.

Há uma CNN+GRU64+Q10 comum por réplica, sem cabeças ou estratégias vencedoras específicas. Ela recebe pixels RGB32×32, ações/recompensas próprias e uma máscara constante dos controles disponíveis. A máscara pode identificar indiretamente o jogo. Os simuladores, as regras, os controles e as recompensas foram preparados por jogo; a rede descobre sua política por experiência. Isso não constitui domínio de jogos arbitrários desconhecidos.

Ocultar pixels ou resetar a memória recorrente altera decisões em todos os jogos e prejudica o desempenho em Snake e Flappy. Desligar o modelo de mundo também altera ações, mas seu benefício de desempenho não foi demonstrado em nenhum dos cinco jogos. Prever estados e alterar decisões não bastam para afirmar que o planejamento aprendido ajudou. A previsão de próximo estado passou em todas as réplicas somente em Snake e Flappy.

## Execução e integridade

- Fonte científica A: `dc71247b2269d41c8d2aeb8c7d489977fac3ad3a`.
- Workflow B: `1c824f17922aa5dfb415e315e7873a1ebfb6db57`.
- Protocolo SHA-256: `93591ede5951438f92432fc6601773aaafe28b2ba5840309aa7bd3611170a324`.
- [Campanha 41551291622, tentativa 1](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41551291622): treino e avaliação concluídos, nove checkpoints, 1.735 episódios completos, 1.080.000 passos de treino + 174.664 de avaliação = 1.254.664. CPU 134,47 minutos; parede 67,72 minutos; pico de memória 690.126.848 bytes. Limites respeitados.
- Artefato científico 13195542205: 50.240.431 bytes, SHA-256 `f396acfcbc8de35e67cbe79be4b6b986d7b536019bca2a62a3effa60ccefa129`. ZIP original e todos os manifestos autenticados; 4.172 arquivos extraídos, 220.570.168 bytes.
- Autorização humana única registrada em #6591517615; [recertificação Chat1](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6591597326); atestado #6591819432 e reserva #6591852632. A autorização foi consumida. A flag estática `execution_authorized=False` continua intacta: as autorizações foram externas e verificadas pelo workflow.

## Recuperação e revisão

[Verificação offline Linux 41619297256, tentativa 1](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41619297256) restaurou exatamente os nove checkpoints: pesos online/target, otimizador, memória recorrente, observações, estados físicos, replay e geradores aleatórios. Não avançou simuladores e não fez atualizações. Reproduziu 962 decisões dos cinco episódios demonstrativos fixados antes da avaliação, com erro numérico máximo zero e tolerância original 1e-6.

Artefato dessa prova: 13222562572, 7.456.919 bytes, SHA-256 `7982426e8965496cda50d789082fb8fe73de4c32a4087006ab90b8a69cd1aeab`. Ambiente original: Python3.11.17/Linux/x86_64, Torch2.10.0+cpu, NumPy2.3.5, Gymnasium1.4.0. A falha anterior no Windows (1,907e-6 contra tolerância1e-6) permanece preservada, sem relaxamento da regra.

CI A41546462545 e B41546841696: 474 testes + 31 subtestes aprovados. O teste de continuidade de 25 atualizações usa replay sintético; não é continuação científica dos modelos oficiais. A recuperação real acima restaura esses modelos sem avançá-los.

Auditorias independentes `code_review_final`, `world_results_audit` e `engineering_receipts` recalcularam os contrastes/ablação ou revisaram fonte, contratos, logs e orçamentos. Os 21 contrastes de competência coincidem com o analisador congelado a 1e-12. Os JSON/Markdown em `audits/` registram escopo e limitações; testes e integridade não substituem desempenho.

| Critério | Estado após revisão técnica, antes da entrega em nuvem |
| --- | --- |
| 1 Snake | PASS |
| 2 Pong | FAIL |
| 3 Agar | FAIL |
| 4 Flappy | FAIL |
| 5 Corrida | FAIL |
| 6 Política compartilhada | PASS |
| 7 Pixels/GRU | PASS |
| 8 Modelo de mundo | FAIL |
| 9 Persistência | PASS, fixture sintética de 25 + restauração real sem avanço |
| 10 Transferência/retenção | FAIL |
| 11 Separação e limites | PASS |
| 12 Entrega e recuperação | Recuperação comprovada; confirmação de nuvem no recibo externo posterior |
| 13 Coordenação final | Decisão final real do Chat1 ainda não recebida; mensagens enviadas não contam como aceite |

Três réplicas e simuladores locais limitam as conclusões. Pong usa somente o oponente assimétrico congelado. Intervalos são exploratórios. Nenhum limiar foi alterado depois de observar os dados, nenhuma réplica foi escolhida por desempenho e nenhuma falha foi descartada.

## Preservação e estado final

O pacote ATUAL deve incluir os ZIPs originais 001 e 002, fonte Git recuperável, protocolo, aprovações, logs, auditorias, resultados, figura descritiva, cinco gravações, trace causal e prova de restauração. O recibo externo, gravado somente após download e conferência do Drive, determina o estado efetivo da entrega. O ZIP original001 SHA-256 `d43466306d14b2d6a811a439f678ec5ceba801e44cdf9f1520cb6bbc899cce32` e seu recibo `cb093b2f5f7545dd12291da0186871512142ade6f74ba114d7d0d73c73b070f3` são preservados byte a byte dentro do novo pacote.

Sem merge e sem mudanças nos checkpoints históricos. TeraBox permanece pendente por decisão explícita do usuário. [PR13](https://github.com/Myrroth/NOVA-Neural-Arena/pull/13) é rascunho; [Issue12](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12) permanece aberta para a visão não alcançada. Novos treinamentos exigiriam uma decisão e autorização novas; este relatório não as solicita nem autoriza.
