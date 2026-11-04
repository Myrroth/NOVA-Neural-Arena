A tentativa003 terminou antes do aprendizado. No [run41758309271, tentativa1](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41758309271), o job de autorização perdeu a confirmação do POST de reserva; o job científico foi ignorado. O log não registrou código HTTP nem categoria da falha. A inexistência observada de um comentário de reserva não prova que seja seguro repetir o POST.

Esta alteração é somente uma correção de diagnóstico para revisão futura. Ela não modifica a fonte científicaA `602fc18cfdfdac6efab51c272b83b9ca3afd7455`, o protocoloP `350f5edef5bc4bbdd86f0c960a8728169612ce4fb279292e9cc83717fb73bbf9` ou o fluxoB `7a31455ee362ed2eb891ec55f1190e2bb48c5131`; também não autoriza outra execução003.

O transporte agora conserva apenas uma categoria de erro de lista fechada e um código HTTP inteiro entre100 e599, quando disponível. Ele distingue erro HTTP, timeout, falha de rede, redirect recusado, status inesperado, JSON inválido e resposta excessiva. Exceções, URLs recebidas, corpos e cabeçalhos não são copiados. O wrapper da reserva preserva esse diagnóstico seguro até o CLI, sem trocar a primeira categoria por outra mensagem genérica. Um código HTTP nos testes é uma resposta inventada, sem atribuir essa causa à tentativa real.

A gravação local passa a ter estes marcos:

- `ATTEMPTED.json` é escrito, sincronizado e tornado visível antes da única chamada POST. Registra missão, repositório, PR, versõesA/B/P, run, tentativa e fase. Esse marco registra entrada na fronteira do POST; não comprova entrega ao GitHub.
- `AUTHORIZATION_STATUS.json` registra `STOPPED` e o diagnóstico seguro em erro. Não contém resposta humana, referência privada ao consentimento, corpo do comentário ou token. Se a própria gravação do resumo falhar, o primeiro erro continua no log e o marco anterior permanece disponível.
- Em sucesso, os mesmos arquivos registram a confirmação, junto dos recibos existentes. O formato e o hash do recibo consumido pelo runner continuam iguais.

Se o marco anterior ao POST não puder ser persistido, o POST não é chamado. As gravações usam um arquivo temporário sincronizado e renomeação; arquivos de diagnóstico existentes não são sobrescritos. Uma falha do POST sem confirmação permanece incerta, exige inspeção e não produz permit confirmado. Se a reserva foi confirmada e depois falhou uma gravação, os arquivos já produzidos são preservados e o estado registra essa diferença; o CLI retorna falha e o workflow impede o job científico, mesmo que um recibo já tenha sido gravado. Nenhum dos erros adiciona retry, resume, fallback de transporte, permissões ou orçamento.

Os testes usam apenas respostas HTTP, eventos e falhas de disco inventados. Cobrem POST201, HTTP403, timeout direto ou encapsulado, redirect, corpo não JSON, status inválido, limite de resposta, retenção da primeira categoria, redaction, ordem da gravação anterior ao POST e preservação do estado em erro. Não fazem requisições reais, jogos, atualização de modelos ou campanhas.

A garantia de artefato depende da disponibilidade do armazenamento e da etapa de upload. A correção evita a pasta vazia causada pela escrita somente após `reserve()` retornar; não promete sobrevivência a perda total do disco, término forçado do processo ou indisponibilidade do serviço de artefatos.
