# Preservação e recuperação das Fases 1–4

Em 01/11/2026 (America/Sao_Paulo), os seis ZIPs originais foram baixados do GitHub Actions. Seus tamanhos e SHA-256 coincidiram com os digests da API e os registros científicos. A recuperação verificou os **38 hashes dos manifestos originais**, além de registrar separadamente três arquivos complementares da Fase 2. Nenhum ZIP, checkpoint ou resultado original foi alterado.

## Cópia persistente

- [Pasta privada no Google Drive](https://drive.google.com/drive/folders/1sOW2MUpeJ7V_bfuH6NxVfqOYbwOUO7gi).
- [Pacote de preservação v1](https://drive.google.com/file/d/1PBfYnWoS3O6oSYlOfTzFJJXtLB0eaYxQ/view): `NOVA-Neural-Arena-fases1-4-preservacao-v1.zip`, **455.899 bytes**.
- SHA-256 independente do pacote: `692f7e1258741a431699f6e7d87c47ab7666a08a94dabeec845dd3a815caed19`.

O upload foi verificado por leitura dos metadados e download completo de retorno. Tamanho e SHA-256 permaneceram idênticos; os **17 hashes do manifesto do pacote** foram recalculados. Os metadados de permissão retornaram somente o proprietário, sem concessão pública. Nenhum link público de compartilhamento foi criado. A persistência depende da manutenção da conta e da pasta pelo proprietário; não é uma garantia de retenção perpétua.

O pacote contém os seis ZIPs internos **byte a byte originais**, um Git bundle com todo o histórico das Fases 1–4, snapshots das discussões/relatórios e APIs de runs/artefatos, inventário de integridade e ferramentas de recuperação em Python sem dependências externas. A recompressão externa não modifica os ZIPs internos. `git bundle verify` confirmou histórico completo e sete referências. Os executáveis congelados e seus commits estão incluídos.

As cópias locais estão em `experiments/archive/original-zips/`, `experiments/archive/verified/`, `experiments/archive/historical-source.bundle` e `experiments/archive/NOVA-Neural-Arena-fases1-4-preservacao-v1.zip`. Esses arquivos são ignorados pelo Git. O download de retorno está em `experiments/archive/drive-roundtrip.zip`. Repositório público recebe apenas documentação, metadados e hashes; modelos não são adicionados ao Git.

## Inventário e limites

[history/preserved-inventory.json](history/preserved-inventory.json) registra nomes, IDs, runs, hashes ZIP/modelos/arquivos, contagens de linhas e expirações originais. [HISTORICAL_AUDIT.md](HISTORICAL_AUDIT.md) identifica os commits e relatórios preservados.

O `SHA256SUMS` original da Fase 2 cobre somente `episodes.csv`, `config.json` e `summary.json`. Seus arquivos de commits e dependências são autenticados pelo digest do ZIP original e têm hashes complementares no inventário. O manifesto original foi preservado exatamente como publicado; não foi ampliado silenciosamente. Os demais cinco manifestos cobrem todos os respectivos arquivos científicos.

A numeração literal dos relatórios 001 e 003 não foi confirmada: os pareceres correspondentes estão preservados com seus títulos originais. Gráficos/CSVs derivados do relatório 002, mencionados como gerados em outro chat, não foram localizados; não estão incluídos. Os dados brutos oficiais necessários para novas análises estão nos ZIPs.

## Procedimento de recuperação

1. Baixe o pacote do Drive, calcule SHA-256 e compare com o valor acima antes de abrir modelos. Use uma pasta nova para cada recuperação.
2. Extraia o pacote em uma pasta privada. Leia `RECOVERY.txt` e confira o manifesto externo.
3. Dentro de `recovery-tools/`, execute em Python ≥3.11:

   ```text
   python -m arena.shared.preservation --zips ../original-zips --out ../recovered --metadata ../history/actions-artifacts-metadata.json
   ```

   O comando verifica todos os seis digests antes de recuperar os arquivos, recusa caminhos inseguros e sobrescritas, verifica os manifestos internos e confere as contagens históricas de episódios. Não carrega modelos nem executa jogos. Uma falha pode deixar uma pasta parcial para inspeção: faça uma nova tentativa em outra pasta, sem apagar a anterior automaticamente.
4. Restaure o código em outra pasta nova:

   ```text
   git clone ../historical-source.bundle restored-source
   git -C restored-source switch -c restored-phase4 8b4b6696e980191349e803218af6e6d55d8ff7e9
   ```

   Os comandos acima continuam dentro de `recovery-tools/`. Cada fase deve usar seu próprio commit congelado documentado na auditoria. O bundle e os helpers empacotados são um snapshot da preservação inicial, anterior à finalização desta PR; as ferramentas atuais de carregamento e proteção adicional de junctions estão na branch desta entrega.
5. Para inferência futura, instale as dependências oficiais em ambiente isolado, confira o `pip-freeze.txt` do artefato e use o novo `arena.shared.checkpoints.load_official_checkpoint` desta PR para auditar/carregar os modelos. A API exige a seed oficial e hashes fixos, verifica os mesmos bytes que desserializa, usa CPU e `weights_only=True`, valida tensores e congela os parâmetros. Os arquivos recuperados estão separados por ID; copie-os para pastas novas `seed_11/`, `seed_42/`, `seed_73/` se utilizar o preparador de políticas da Fase 5.

Python 3.11.17 foi usado no piloto oficial; a versão local disponível é 3.11.0. Dependências diretas: CPU torch 2.10.0, NumPy 2.3.5, Gymnasium 1.4.0, pytest 9.0.2. Versões transitivas efetivas estão em cada ZIP. Pesos recuperados permitem restaurar inferência, mas os checkpoints antigos não contêm replay/RNG suficientes para retomar treinamento de forma idêntica.

Recuperar, conferir e carregar não autoriza repetir campanhas ou iniciar treinamentos. A Fase 5 continua dependente da decisão expressa do Chat 1. Futuras revisões do pacote devem receber nome e hash novos; nunca substituir silenciosamente a versão v1.
