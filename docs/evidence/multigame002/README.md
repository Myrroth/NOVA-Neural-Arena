# G1: medição técnica e derivação preservadas

Estes arquivos não contêm resultados de aprendizado. A execução técnica
[41500156977, tentativa 1](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41500156977)
rodou os testes completos (326 testes e 31 subtestes aprovados), 475 passos
mecânicos aleatórios e 100 atualizações sobre ruído sintético. Nenhuma
observação de jogo foi usada para otimização e nenhum modelo científico foi
criado. O job falhou na validação final dos contadores de memória.

O arquivo `preflight-original-failed.json` conserva os bytes e o estado FAILED
do receipt original. A comparação RSS atual <= pico em cada amostra era
incorreta para leituras independentes de `/proc/self/statm` e `getrusage`:
duas leituras diferem em 204.800 e 151.552 bytes. Os quatro registros são
preservados. A [documentação do contador Linux](https://man7.org/linux/man-pages/man5/proc_pid_statm.5.html)
identifica RSS como aproximado; os provedores não são lidos atomicamente.

`preflight-offline-derivation.json` é uma DERIVAÇÃO, não nova medição e não
um PASS. Ela autentica os bytes originais, exige a carga completa e hashes
dos cinco módulos relevantes idênticos aos medidos, conserva todos os
tempos/contadores e usa o máximo de todos os RSS atuais e picos observados.
Não executa jogos ou atualizações e não reclassifica a falha original.

Projeção com margem de 20%: 75,19 minutos de tempo decorrido, 155,80 minutos
de CPU de processo e 1,23 GiB de RSS. São estimativas do job Linux medido,
não garantia para qualquer hardware. O aceite G1 dessa derivação continua
pendente de decisão explícita do Chat 1. O limite interno de execução
monitora separadamente tempo decorrido e CPU acumulada e encerra em 285
minutos, preservando margem para finalizar os artefatos dentro de 300 minutos.

SHA256 dos bytes preservados:

| Arquivo | SHA256 |
| --- | --- |
| ZIP técnico original (12.550 bytes, mantido em experiments) | `72dd6cddf504d2192532dfea50138e22f7779032d23ca456179f1c5e093809a0` |
| preflight-original-failed.json | `aa7ee8964afa6f412eec308c6dba6a8763b82cdcfe72cc3fb74828c956ecc97d` |
| preflight-offline-derivation.json | `ee84f06e7d8754feae6e032f95ce4dcdcbc5bf2124fc8948328254a9efcdea94` |

O workflow original fez checkout do merge virtual
`bd447baa601dd4ef5dfe18e9932ac75e3dc313da`, a partir do HEAD
`0538898f419b4618708071d143de4cf4ee6f509f`. Os hashes individuais estão nos
receipts. O workflow de engenharia agora fixa explicitamente o HEAD da PR.

O [Chat1 aceitou G1 com ressalvas](https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6590543972),
sem reclassificar o job original ou a derivação em PASS. A fonte seguinte
moveu a mesma chamada preditiva da avaliação para antes de env.step: mesmos
argumentos, operadores e número de chamadas; nenhum jogo/update repetido.
Essa diferença textual em evaluation.py precisa da recertificação final de
equivalência de custo, preservando a medição e derivação originais intocadas.
Os outros quatro módulos de desempenho continuam idênticos aos medidos.

Nenhum destes documentos concede autorização científica. Fonte, protocolo,
revisões e consentimento humano novo continuam requisitos separados.
