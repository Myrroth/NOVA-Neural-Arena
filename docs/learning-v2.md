# Correções do aprendizado — nova linhagem v2

O usuário pediu expressamente corrigir o aprendizado do projeto após o encerramento da campanha002. Esta versão está em `arena/continual/`; `arena/universal/`, os ambientes históricos, os checkpoints e os resultados001/002 permanecem intactos. Não há merge nem campanha científica disparada por esta mudança.

## Problema e comportamento novo

| Problema observado no mecanismo | Correção implementada | Prova técnica |
| --- | --- | --- |
| Amostrar jogos em proporções definidas não bastava: uma sequência com8passos válidos tinha8vezes o peso de outra com1passo no denominador global. | A perda é normalizada dentro de cada sequência antes da média do lote. | Fixture numérica com comprimentos diferentes e mesma contribuição por sequência. |
| O estado recorrente do ator era reutilizado depois de mudanças no encoder/GRU. | Contextos de até16quadros, com reset periódico explícito e cache vinculado à revisão dos parâmetros. Após mudança, o contexto é reconstruído com os pesos atuais. | Comparação com histórico reconstruído e um controlador recorrente independente, inclusive depois de evicção de prefixo desalinhado. |
| O modelo de mundo só podia prever alteração de até0,1 por coordenada, embora o estado seguinte pudesse mudar muito mais. | Amplitude residual2, cobrindo a faixa dos estados da GRU; inicialização ainda prevê cópia. | Transição sintética antes irrepresentável e teste da inicialização. |
| A perda de mundo podia mover a representação usada pelo controle. | Estados atuais/sucessores detached e otimizador exclusivo da transição. | Update de mundo altera a transição e deixa encoder/GRU/Q e otimizador de controle intactos. |
| Não havia objetivo explícito de preservar a política fonte durante adaptação. | Professor congelado do próprio agente e cópia protegida de suas experiências-fonte; KL apenas sobre ações legais. Professor e estudante reconstruem seus próprios estados. | Professor imutável, replay fonte não recebe alvo e teste de gradientes/máscaras da destilação. |
| Mistura fixa podia usar previsões não validadas. | Planejamento começa desligado. Episódios de calibração reservados não entram nas perdas de ajuste; suporte é contado por transições distintas, no domínio ativo e na revisão atual. | Casos frio, ruim, duplicado, desatualizado, sem alvo reservado e após mutação do mundo deixam o gate desligado. |
| Crédito de recompensa era de um passo. | Retorno3passos, preservando a diferença entre término físico e truncamento. | Alvos calculados manualmente em término, limite externo, padding e horizonte. |

Esses são mecanismos verificáveis e hipóteses de melhoria. Não estabelecem que cada um causou os seis FAIL históricos, nem que a nova versão já aprendeu os cinco jogos.

## Memória, calibração e persistência

O contexto periódico é um limite novo e explícito: não conserva informação anterior aos16quadros do bloco. O replay só sorteia âncoras cujo reset de contexto ainda está presente; se o buffer não puder fornecer contexto completo, não inventa estado inicial nem muda silenciosamente os pesos dos jogos.

O encoder processa os quadros independentes do lote de uma vez; a GRU continua sequencial e respeita os resets. O encoder atual não contém BatchNorm nem dropout. Testes confrontam saídas e gradientes com o laço original e verificam que quadros futuros não alteram estados anteriores. Isso reduz trabalho repetido sem fornecer observações futuras ao controle.

A cada10episódios, um episódio inteiro é reservado para calibração. As sementes de ajuste são pares, as de calibração ímpares, em uma nova família acima de2.000.000. Isso garante separação dos fluxos; adaptação e controle do zero usam o mesmo fluxo ambiental do alvo. As sementes finais002 não são usadas nesta implementação ou nos testes novos.

A calibração verifica erro de próximo estado contra cópia, recompensa contra média e término contra taxa observada. Cada ação legal precisa de suporte mínimo de8transições distintas. Não há ID do jogo na rede ou no gate; os nomes só organizam dados/estágios. Uma boa calibração é estimativa local em dados de treinamento, não garantia em qualquer estado novo ou prova de benefício causal. Mudanças nos pesos do controle invalidam a revisão; mudanças no mundo desligam a confiança antes da atualização. Nenhuma calibração fonte autoriza o primeiro uso no alvo ainda desconhecido.

O snapshot v2 inclui política/target/professor, dois otimizadores, replay de ajuste, replay de calibração, replay fonte protegido, geradores aleatórios, estado físico, histórico de observações, contadores e gate. A carga exige fonte/protocolo/arquitetura/runtime e manifesto confiáveis, usa `weights_only=True` e recusa checkpoints históricos v1. Cache de estados recorrentes não é persistido: é reconstruído do histórico autenticado. A curva registra episódios parciais no extremo exato do orçamento, sem duplicar registros em uma coleta de0passos.

## Validação e limites

Os testes novos usam tensores, pixels artificiais e atores sintéticos. Incluem continuidade de25atualizações sintéticas após salvar/carregar, perda equilibrada, isolamento de gradientes, professor imutável, suporte de calibração, bootstrap e segurança do snapshot. A suíte completa também continua verificando o código histórico. Esses testes não são partidas científicas e não comprovam melhoria de competência.

Não há controlador vencedor por jogo, ações de especialista, head específico, modificação de recompensa dos modelos antigos ou limiar científico relaxado. O decoder herdado está congelado e não é apresentado como reconstrução aprendida. A mistura, a destilação e o retorno3passos precisam ser avaliados por experimento novo com controles e dados finais não usados no diagnóstico002.

A autorização singular002 foi consumida. Sua decisão de7PASS/6FAIL e o pacote no Drive continuam válidos. A CI desta branch executa somente verificações de engenharia. Não há permissão implícita para uma campanha, repetição, gasto pago ou merge. Um teste científico da nova versão exige protocolo/versão/fontes/budget próprios e autorização nova, sem reaproveitar a aprovação002.

## Avaliação preparada e custo ainda pendente

O [plano finito003](learning-v2-validation.md) descreve três réplicas e1.735episódios finais, com as margens científicas anteriores. O avaliador copia o histórico do controlador para contrafactuais, registra a mistura realmente usada e faz previsões antes da interação. A carga para inferência autentica o snapshot inteiro sem restaurar replay, atores ou otimizadores. O analisador exige a nova linhagem e cobertura de todas as transições.

O runner mantém reserva local exclusiva, contadores por interação e preservação de falhas, inclusive quando um registro não pode ser exportado. Contabiliza oportunidades, tentativas, atualizações e skips; a comparação entre adaptação e scratch usa o mesmo orçamento ambiental e publica diferenças de custo. Sua execução de produção exige protocolo fixado, consentimento humano novo, parecer real e recibo da reserva remota autenticado pelo job separado; um JSON aprovado localmente não basta.

Foram feitos dois perfis finitos em tensores aleatórios, sem partidas, modelos históricos ou sementes finais. O primeiro identificou repetição do encoder; o segundo mediu a correção equivalente. Os [dados no Windows](evidence/learning-v2/synthetic-cost-context16-vectorized-windows.json) ainda não demonstram encaixe em300minutos: só a extrapolação dos otimizadores seria de cerca de367minutos de execução e660minutos de CPU neste computador. Isso não estima o desempenho do runner Linux. A medição única no Linux terminou no run41752386162:100atualizações e100operações sintéticas, sem jogos. A [avaliação independente](evidence/learning-v2/linux-cost-assessment.json) levou à proposta explícita de360minutos decorridos/720minutosCPU, com paradas330/660 e margens modeladas. Essa proposta continua aguardando os consentimentos necessários; a medição não autorizou campanha ou nova tentativa automática.

## Fundamentação

O cuidado com deriva de representações e estado recorrente obsoleto é discutido no artigo [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX). A proteção por destilação adapta um princípio de [Learning without Forgetting](https://arxiv.org/abs/1606.09282); aqui há experiências próprias protegidas e uma política de reforço, portanto não se trata de reprodução daquele método de classificação. A distinção de bootstrap segue a [API oficial do Gymnasium](https://farama.org/Gymnasium-Terminated-Truncated-Step-API). Nenhuma dessas referências garante o resultado deste projeto.
