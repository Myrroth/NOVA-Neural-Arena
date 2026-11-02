# Revisões independentes antes da ciência

Fonte examinada: `2de3d4acfde89df1bcc523a52bd41b377e77d0f5`.
CI dessa fonte: [41542183123](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41542183123),
465 testes e 31 subtestes aprovados, dois avisos informativos.

Dois subagentes fizeram revisão somente leitura, sem jogos, gradientes,
benchmarks, uso das sementes finais ou alterações nos arquivos. São revisores
independentes desta implementação, **não são os chats externos 2, 3 e 4** e
seus pareceres não representam aceite daqueles chats ou do Chat1.

## Política, replay, ambientes e persistência

Revisor `code_review_final`: nenhum bloqueador material identificado.
Conferiu alinhamento temporal t/t+1, DoubleDQN com argmax online/valor target,
sucessor do mundo ONLINE.detach, EMA somente no TD, padding fora da perda,
bootstrap preservado na truncação e previsão antes de env.step. A política
recebe pixels, controles mecânicos e história própria; não foram identificadas
features privilegiadas ou futuras.

Achado P3 documental: protocolo dizia NumPy default_rng para o replay, enquanto
o código usa Python random.Random(model_seed+71), com getstate completo salvo
e restaurado. A descrição foi corrigida antes do congelamento, sem alterar
o gerador, as sementes ou a implementação de aprendizado.

## Reserva, limites e análise

Revisor `analysis_review_final`: nenhum bloqueador novo de reserva ou
orquestração identificado. Conferiu paginação completa de comentários,
aprovação única e recertificação literal, POST único sem retry, vínculo de
fonte e workflow, bloqueio de protocolo candidato, orçamento acumulado,
bootstrap, retenção, métricas do mundo e distinção entre mudança de decisão
e benefício no resultado.

Achado P2 de integração offline: os três CSVs da avaliação ficam em
`evaluation/`, enquanto `training_curve.csv` fica na raiz da campanha.
O analisador buscava todos no mesmo diretório. A correção permite apontar
explicitamente a curva externa com SHA256 obrigatório, autenticando os mesmos
bytes lidos e preservando os originais. Não altera estimandos, fórmulas,
limiares ou matriz; não requer coleta adicional.

STOP285 é sondado entre chunks finitos e episódios, com margem até300min;
essa granularidade permanece limitação de execução, não garantia de duração.

## Alcance e próximos aceites

Revisão estática e testes de software não demonstram aprendizado. Os treze
critérios científicos continuam sem comprovação. Pareceres externos dos chats
2/3/4, recertificação do Chat1, fonte/hash congelados e consentimento humano
novo continuam etapas distintas. Nenhuma campanha científica ou reserva foi
autorizada por estes relatórios.

`pre-freeze-static-cost.json` conserva a comparação estática das chamadas da
avaliação antes/depois da mudança de ordem. O multiconjunto de chamadas AST
é idêntico; isso não é benchmark ou prova matemática de desempenho.
Uma análise de sensibilidade cobra quatro unidades completas de inferência
e mecânica por passo de toda a avaliação, além das atualizações máximas e
margem20%: 110,24min wall e232,39min CPU. É estimativa, não nova medição nem
garantia; não cobre IO externo/checkpoints. A aplicabilidade final depende da
recertificação de custo do Chat1. Nenhum passo ou update foi repetido.
