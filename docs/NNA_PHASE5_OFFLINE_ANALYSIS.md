# NNA-FECHAMENTO-001 — plano de análise OFFLINE da Fase 5

**Estado: INSTRUMENTO PREPARADO, SEM DADOS OFICIAIS. NÃO HÁ CAMPANHA EXECUTADA.**

O Work informou que **a revisão automática bloqueou o encaminhamento dos
32.200 episódios** e exige uma autorização direta e explícita do proprietário.
Até essa resposta ser recebida e comunicada, não aplicar etiqueta, criar reserva,
disparar workflow ou iniciar partida. A decisão científica do Chat 1 está na
[Issue #6, comentário 6568072221](https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-6568072221);
ela **não supera o bloqueio adicional**. O Work é o responsável pelo contato
direto com o proprietário e com os Chats 3/4.

Este módulo é separado do executor oficial congelado em
\`39461168db30edb90a25385fb3a1eaca8dfdde33\`, e **não deve alterar**
\`phase5_campaign.py\`, \`phase5_hardstates.py\`, o workflow autorizado, Snake,
pesos, nem checkpoints antigos.

## Contrato de entrada

O analisador \`arena.analysis.phase5_offline\` usa somente a biblioteca padrão
do Python. Nunca importa PyTorch, Gymnasium, simulador ou modelo.

**Ele só aceita um pacote científico previamente existente**, com:

- \`protocol.json\`, \`execution.json\`, \`pip-freeze.txt\`,
  \`episodes.csv\`, \`status.json\`, \`SHA256SUMS\`.
- Manifesto protocolado: \`phase5-hardstates-closure-approved-v1\`,
  SHA-256 **\`ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a\`**.
- Código congelado indicado como
  **\`39461168db30edb90a25385fb3a1eaca8dfdde33\`**.
- Estado completo: 32.200 episódios, 92 cenários, políticas 7 e sementes
  500001–500050, no máximo 200 passos por episódio e 6.440.000 no total.
- Comprovantes de run/attempt/reserva, ausência de treino, integridade dos
  pesos verificada ao terminar, e hashes SHA256 internos dos cinco arquivos.
- **Hash SHA-256 de \`episodes.csv\` obtido de fonte oficial independente**,
  separado do próprio ZIP/manifesto, que deve ser conferido com os registros
  do Chat 3/4 e GitHub Actions. Um manifesto distribuído junto de arquivo
  adulterado, sozinho, não autentica sua origem. Também conferir o digest do
  ZIP oficial na fonte GitHub antes de extrair.

**Não aceite como relatório científico um arquivo sintético de testes.** Os
testes desta PR criam dados CSV completamente fictícios, sem qualquer episódio
do simulador, e verificam apenas a matemática e segurança dos parsers.

## Comando RESTRITO A DEPOIS de existir o artefato oficial e autorização

\`\`\`bash
python -m arena.analysis.phase5_offline \
    --source /private/phase5/extracted-official-result \
    --out /private/phase5/offline-analysis-NEW \
    --expected-csv-sha256 SHA256_INDEPENDENTE_PUBLICADO_PELO_CHAT3
\`\`\`

O comando **não** executa fases, não carrega modelos e não altera os arquivos
da campanha. O diretório de saída deve ser novo; uma segunda invocação no mesmo
diretório é recusada para evitar sobrescrita.

## Estatística pré-registrada

- **Primário:** alimentos por partida (\`score\`), reportados separadamente
  para as três DQNs 11/42/73.
- **Seis contrastes primários**, DQN treinada menos seu controle não treinado
  correspondente e menos a política aleatória. Parear \`scenario_id\` e
  \`environment_seed\`, fazer média balanceada de 92 cenários para **cada uma das
  50 sementes** e então média dos 50 blocos.
- **Incerteza:** bootstrap percentil 95%, 10.000 reamostragens, semente 510000.
  Cada amostragem seleciona **o bloco da semente ambiental completo**, mantendo
  intactas as 92 condições/7 políticas e suas dependências. Mesmos índices
  bootstrap reutilizados entre os seis contrastes.
- **Secundários:** 552 comparações treinado-vs-controle por cenário e 216
  contrastes difícil-vs-centro, com a **mesma direção, \`food_mode\` e
  \`food_key\`**. Somente estimativas descritivas, não são testes simultâneos.
- Diagnósticos: taxa de coletar pelo menos um alimento, censura de primeira
  comida, colisão, limite de passos, repetição do estado físico completo
  (conforme contagem registrada no CSV), e número de pontuações diferentes.
  **Pontuações diferentes não equivalem a trajetórias únicas**; determinar
  trajetórias únicas exatas exige ações/estados completos preservados.
- **Não realizar testes confirmatórios**, emitir p-valores, interpretar ausência
  de significância como equivalência, ou inventar margem de não inferioridade.
  Bootstrap é condicional aos três checkpoints fixos; 50 seeds não são 50
  treinamentos independentes.

## Saídas derivadas

- \`analysis.json\`: seis contrastes, intervalos bootstrap e proveniência.
- \`policy-vs-controls.csv\`: 552 estimativas secundárias.
- \`difficult-vs-center.csv\`: 216 estimativas secundárias.
- \`trained-diagnostics.csv\`: métricas por checkpoint/cenário.
- \`SHA256SUMS\`: integridade dos quatro derivados.

Antes de qualquer publicação científica, o Chat 4 confere os resultados com
as fontes brutas, relata cenários com dados repetidos ou censurados e publica o
relatório no GitHub e comunica o link ao Chat 1 na mesma entrega. O Chat 1 faz
a revisão final e presta contas ao Work. A análise offline é apoio técnico,
não substitui a auditoria independente do Chat 4.

**Regra final:** nenhuma execução, nova campanha, treino, merge ou pacote
redundante foi autorizada por este documento.
