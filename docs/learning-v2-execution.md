# Execução única proposta da NNA-CONTINUAL-003

O código de reserva está preparado para revisão. Não há autorização científica nova, registro real de aprovação, reserva remota ou campanha criada por esta implementação. Os testes usam exclusivamente eventos e respostas inventadas do GitHub. O protocolo está fixado para revisão e continua sem autorização de execução. A entrada de produção exige os dois recebimentos reais, a reserva e o recibo do job confiável.

## Fonte e fluxo imutáveis

O commit **A** contém aprendiz, runner, guardas e protocolo **P**. O commit **B** é descendente de A e pode alterar somente `.github/workflows/nna-continual003-once.yml`. A comparação dos dois commits precisa confirmar A como ancestral e a única diferença de arquivo. A fonte científica executada será A; o fluxo revisado e atestado será B.

A única entrada aceita é a etiqueta `nna-continual003-campaign-approved` na PR14 aberta e sem merge, de `fix/nna-learning-v2` para `feat/nna-multigame-002`, no repositório público `Myrroth/NOVA-Neural-Arena`. O remetente precisa ser `Myrroth`, ID288351784. O evento e a consulta atual da PR precisam concordar com B, as branches e a ausência de fork. Só a tentativa1 do run pode prosseguir. O estado de rascunho é permitido.

## Consentimento, parecer e atestado

O operador precisa receber primeiro consentimento humano novo e específico para a campanha003 e um parecer real do Planejamento sobre A, B e P. As aprovações consumidas001/002 não atendem a esse requisito. A autorização pertence à conversa humana; um JSON produzido pelo código não a cria.

Depois desses dois recebimentos reais, o atestado do proprietário deve ser um comentário na própria PR14, com metadados `OWNER`, login e ID exatos. Sua primeira linha é `NNA-CONTINUAL-003:APPROVAL`, seguida de um único objeto JSON sem chaves duplicadas. Seus campos exatos são:

- `approval_type`: `new-explicit-continual-003`.
- `approved`: booleano verdadeiro, somente depois do consentimento real.
- `workflow_head`: SHA completo de B.
- `source_commit`: SHA completo de A.
- `protocol_sha256`: hash completo P.
- `human_approval_reference`: identificador específico da nova mensagem humana; referências antigas ou provisórias são recusadas.
- `planning_approval_url`: URL exata `https://github.com/Myrroth/NOVA-Neural-Arena/pull/14#issuecomment-ID` do parecer novo.
- `budgets`: todos os campos e tipos retornados por `workflow_guard.approval_budget()`, iguais ao protocolo fixado. Incluem os limites distintos de CPU e tempo decorrido, suas paradas internas, passos, updates, memória, uma tentativa, ausência de repetição e recursos pagos.

O comentário do Planejamento referenciado deve pertencer à PR14, ter os metadados exatos do proprietário e começar literalmente com:

```text
NNA-CONTINUAL-003:RECERTIFIED | source A | protocol P | workflow B
```

Nesse exemplo, A/B/P são placeholders deliberadamente inválidos e precisam ser substituídos pelos identificadores completos aprovados. Não se deve publicar esse exemplo como parecer. A linha literal vincula o registro revisado; a API não autentica por si só qual chat produziu seu conteúdo. O operador continua responsável por registrar um parecer que efetivamente recebeu, sem inventar confirmação.

## Reserva remota antes do aprendiz

O workflow precisa manter um grupo global de concorrência exclusivo para a campanha003, com `cancel-in-progress: false`, abrangendo toda a execução. Não pode haver matriz, retry, resume, `workflow_dispatch` ou caminho alternativo que reinicie a campanha.

O job de reserva usa apenas biblioteca padrão e código B já revisado. Ele consulta o repositório, a PR atual, a comparação A...B e todas as páginas de comentários da PR14. Exige exatamente um atestado que corresponda a A/B/P, busca diretamente o comentário do parecer e verifica sua linha literal. O CLI também recusa protocolo candidato, pendências de congelamento ou hash P divergente antes de criar diretório ou acessar a API.

Qualquer comentário anterior que comece com `NNA-CONTINUAL-003:RESERVED | immutable run ` consome a campanha inteira, inclusive se corresponder a outra fonte, execução interrompida ou ao run atual. Menções narrativas desse texto em outro ponto do comentário não são reservas. Com o grupo de concorrência ainda mantido, o job faz um único POST e exige resposta com ID positivo, localização e corpo exatos.

Essa sequência depende da serialização do workflow: a API de comentários não oferece transação entre leitura e POST. Chamadas manuais simultâneas fora do workflow não pertencem ao contrato. Se a resposta do POST se perder ou for ambígua, a reserva pode já existir. O job para e **não repete o POST**. Uma investigação posterior pode ler o estado; não recebe autorização de executar novamente.

## Credenciais e transporte

O job de reserva precisa de `contents: read` e `issues: write`. O token de escrita só é exposto à etapa de reserva; dependências do aprendiz e código A não são executados nesse job. O job científico separado usa `contents: read`, recebe o artefato da reserva e faz checkout de A. Não recebe o token de escrita do job anterior.

O CLI de reserva recebe:

```text
python -m arena.continual.workflow_reservation --expected-source A --expected-workflow B --expected-protocol P --output NOVO_DIRETORIO_EXTERNO_AO_GIT
```

`GITHUB_EVENT_PATH`, `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT` e `GH_TOKEN` são fornecidos pelo job confiável. O transporte HTTP aceita somente endpoints HTTPS fixos do repositório, não segue redirects e não tenta novamente requisições. Erros de transporte não incluem tokens ou respostas privadas nos logs.

O diretório de saída exclusivo contém `receipt.json`, `permit.json` e `SHA256SUMS`. Os JSONs têm serialização UTF8 canônica, incluindo newline final. O digest do recibo cobre seus bytes exatos e também sai em `GITHUB_OUTPUT` como `receipt_sha256`, junto com `reservation_id`. O digest esperado pelo job científico deve vir desse output confiável, não ser recalculado a partir de um permit editável.

O permit vincula `source_revision`, `workflow_revision`, `protocol_sha256`, `workflow_run_id`, `workflow_run_attempt`, `human_approval_reference`, `planning_approval_reference`, `reservation_id`, `reservation_receipt_sha256` e `budgets`. `authorization_type` precisa ser `new-explicit-continual-003` e `approved` precisa ser booleano verdadeiro. O recibo usa schema2, missão003, repositório e PR14, além das identidades dos comentários e do texto exato da reserva.

O runner deve chamar `read_and_validate_receipt(...)` antes de criar aprendiz ou fazer interação, passando o SHA esperado do job anterior e A/B/P/run/tentativa esperados. O helper lê somente dois arquivos JSON pequenos e comuns, recusa links/reparse points, formatos não canônicos e qualquer divergência. Não importa Torch/Gym, não consulta API e não usa credenciais.

## Limites da prova

As verificações comprovam integridade e vínculo do transporte dentro do workflow revisado. Não comprovam competência nos jogos, benefício do mundo, retenção, verdade de uma conversa humana ou segurança contra um operador privilegiado que substitua o próprio fluxo. Essas conclusões exigem as evidências e controles do protocolo científico.

Uma reserva confirmada não amplia orçamento nem permite ultrapassar os limites. Depois da única tentativa, sucesso operacional, resultado negativo, falha de gravação, timeout ou interrupção encerram essa execução. O pacote disponível e os resultados incompletos devem ser preservados sem reinício automático.
