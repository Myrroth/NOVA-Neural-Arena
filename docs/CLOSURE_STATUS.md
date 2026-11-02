# Fechamento NNA-FECHAMENTO-001

Registro de 02/11/2026. A campanha única da Fase 5 terminou com sucesso; a
engenharia foi entregue para revisão na [PR #11](https://github.com/Myrroth/NOVA-Neural-Arena/pull/11),
sem merge. Nenhum treinamento adicional ou repetição é parte deste fechamento.
O recibo de publicação do pacote ATUAL, produzido depois deste commit, fica na
[Issue #6](https://github.com/Myrroth/NOVA-Neural-Arena/issues/6).

## Avaliação científica concluída

- [Run 41435760871](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41435760871),
  tentativa 1, sucesso, reserva efetiva [6575822734](https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-6575822734).
- Artifact **13135514208**, ZIP original de **320.201 bytes**, SHA-256
  `6ed4ebd67c318d4267204dafb0db656be1688ad7866e9b2f5c2b91c9c31dedbe`.
- **32.200 chaves únicas**: 92 condições × 50 seeds ambientais × 7 políticas;
  **3.212.125 passos**, abaixo do teto 6.440.000; até 200 por episódio.
- Cinco hashes internos aprovados; CSV SHA-256
  `10cac710894a79424271bca43f75e2ca6a3595ae97ddba27290d3118b94d1d97`.
- Executor `39461168db30edb90a25385fb3a1eaca8dfdde33`, fonte de operação
  recertificada `28ed5a339e7a14006f0d7fa556dc6bf23821fc10`, protocolo canônico
  `ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a`.
- Pesos congelados e fingerprints antes/depois iguais, `no_training=true` e
  `complete=true`. Os três modelos oficiais e os seis ZIPs das Fases 1–4 são
  preservados byte a byte. A auditoria offline não jogou nem carregou modelos.

O usuário autorizou diretamente essa avaliação única. O Work/0 assumiu como
operador exclusivo após a transferência documentada ao Planejamento; Chat 1 e
Chat 3 receberam STOP para disparos. O evento já foi consumido. Os runs
41383420458 e 41385287550 falharam no pré-voo, com zero episódios e nenhuma
reserva real; os dois incidentes e seus relatórios continuam no histórico.
Isso não autoriza reparar ou disparar outra campanha depois do sucesso.

## Resultado e limites

O [relatório final do Chat 4](https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-6576086043)
e sua [entrega verificada](https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-6576109322)
contêm seis contrastes primários, 552 comparações por condição, 216
difíceis-centro, tabelas, scripts e gráficos feitos dos dados reais.

| Checkpoint | Vantagem sobre não treinada | Vantagem sobre aleatória |
|---|---:|---:|
| 11 | +9,211739 alimentos | +9,122826 alimentos |
| 42 | +14,748913 alimentos | +14,678478 alimentos |
| 73 | +12,805652 alimentos | +12,786087 alimentos |

As médias são balanceadas entre as 92 condições. O relatório usa ICs percentis
de 95%, 10.000 reamostragens de blocos completos de seed ambiental e
`numpy.random.default_rng(510000)`. A auditoria independente do Work usa
`random.Random(510000)`, preservado da ferramenta pré-dados da PR #10. As
estimativas pontuais coincidem; pequenas diferenças Monte Carlo nos limites
decorrem dos motores distintos. Ambos os resultados e métodos são preservados;
não se escolheu o intervalo mais favorável. Sem testes confirmatórios,
p-valores, Holm, equivalência ou não inferioridade.

Quatro pares checkpoint/condição ficaram sem primeira comida em **50/50**
episódios, todos em parede/comida relativa e truncados em 200 passos:

| Checkpoint | Condição |
|---|---|
| 11 | `5,0-north-relative-0,-1` |
| 42 | `9,5-east-relative-0,-1` |
| 73 | `5,9-east-relative-1,0` |
| 73 | `9,5-east-relative-0,-1` |

Há somente três treinamentos independentes. A primeira comida fixa e as
políticas determinísticas reduzem a diversidade pré-coleta; 50 seeds não
equivalem a 50 trajetórias independentes nesse trecho. O CSV não contém hash
de trajetórias, portanto sua quantidade exata é indisponível. Tempos até
primeira comida são apresentados com denominadores e causas de falha; não se
assume censura independente para colisões. A vantagem média não demonstra
robustez universal ou transferência para Pong.

## Engenharia entregue

O Pong original da PR #7 é preservado, inclusive a limitação de recompensa
positiva inalcançável na configuração padrão. A correção isolada
**`pong-local-v2`** faz o adversário reagir a cada dois passos, preserva a
recompensa e a observação, e rejeita contratos de checkpoint v1. Testes partem
do reset, usam observações públicas e ações legais e alcançam +1; demonstram
alcançabilidade, sem alegar aprendizagem. Nenhuma rede foi treinada em Pong.

A [CI final 41434661636](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41434661636),
sobre `fd6f120dc83e99c890c1fff5257bcef7763962e5`, aprovou **203 testes e
31 subtestes**, dois avisos informativos de Gymnasium. Inclui suíte completa,
Pong v2, filtro de reserva, construtor/recuperador do pacote e comparação dos
sete executáveis históricos. Os commits documentais posteriores não alteram
esses executáveis. A CI científica separada aprovou 188 testes no HEAD
recertificado da PR #8.

O analisador offline da PR #10 supunha uma ordem diferente de políticas no
manifesto. A auditoria independente usa a ordem autenticada do executor,
preserva o script derivado e documenta o ajuste; não modifica dados ou protocolo.
Os métodos matemáticos foram cotejados sem executar jogos.

## Preservação e encerramento operacional

O [pacote ATUAL](CURRENT_PACKAGE.md) inclui sete ZIPs científicos originais,
fonte aprovada, fonte final, bundle Git, relatório completo do Chat 4,
auditoria independente, evidências e as cópias anteriores integrais.
Inventário e SHA256SUMS autenticam todos os membros; o SHA externo e o recibo
de retorno do Drive ficam fora do ZIP, na Issue #6, evitando autorreferência.

A publicação exige upload, download de retorno, conferência integral e
recuperação em pasta nova. Só então podem ser retiradas as cópias antigas
integralmente cobertas pelo novo arquivo. O recibo lista os IDs efetivamente
tratados; este documento não presume ações de armazenamento ainda não feitas.

**TeraBox permanece pendente por decisão explícita do usuário.** Não bloqueia
o fechamento no Google Drive. Nenhuma agenda, monitor de treino, novo ciclo
de avaliação, modificação da NOVA original ou merge foi criado pelo fechamento.
