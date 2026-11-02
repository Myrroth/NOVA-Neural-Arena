# Pacote ATUAL — NNA-FECHAMENTO-001

`NOVA-Neural-Arena-ATUAL.zip` é a consolidação canônica do ciclo. Os ZIPs
científicos originais são membros internos preservados byte a byte. O pacote
não retreina, carrega modelos ou executa episódios. A substituição das cópias
antigas nos serviços de armazenamento é uma operação posterior, condicionada
ao upload, download de retorno, hashes e recuperação verificados.

## Conteúdo e identidade

- `scientific/original-zips/`: os seis ZIPs oficiais das Fases 1–4 e o ZIP da
  única campanha da Fase 5; checkpoints 11/42/73 continuam nos ZIPs originais.
- `scientific/final-analysis/`: relatório final, tabelas, gráficos e scripts
  entregues pelo Chat 4. Não confundir esses derivados com os arquivos originais.
- `source/source-approved-*.tar`: fonte aprovada, incluindo Pong e executor da
  Fase 5 no commit `8468e0dad3a75ecfe06187d8dbbf7a523f468df0`.
- `source/source-current-*.tar`: fonte e documentação do commit final indicado
  no inventário; `source/full-source-history.bundle` conserva todas as refs e
  o histórico Git presentes no clone local no momento da consolidação.
- `evidence/`: decisões, snapshots oficiais e evidência de verificação do bundle.
- `recovery-tools/`: verificador do pacote e módulos de integridade necessários,
  usando somente a biblioteca padrão do Python.
- `INVENTORY.json`: commits, protocolo, execução única, arquivos, tamanhos,
  hashes e limitações. `SHA256SUMS` cobre todos os membros, exceto ele próprio.

O checksum externo do ZIP fica em `NOVA-Neural-Arena-ATUAL.zip.sha256` e no
recibo de entrega. Não pode ficar dentro do próprio ZIP como se autenticasse a
si mesmo. Compare-o com um registro independente do GitHub antes de recuperar.

## Construção controlada

O comando deve ser executado apenas depois da campanha única e da entrega do
relatório científico, com a fonte final já registrada em commit e sem alterações
em arquivos rastreados. Saídas antigas nunca são substituídas pelo script.

```text
python scripts/build_current_package.py --inputs experiments/closure/inputs/package-inputs.json --out experiments/closure/final
```

O JSON de entrada tem esta estrutura (substitua os dados da campanha pelos
identificadores e hashes efetivamente conferidos; não invente valores):

```json
{
  "schema_version": 1,
  "mission": "NNA-FECHAMENTO-001",
  "approved_source_commit": "8468e0dad3a75ecfe06187d8dbbf7a523f468df0",
  "final_source_commit": "SHA_COMPLETO_DO_HEAD_FINAL",
  "scientific_executor_commit": "39461168db30edb90a25385fb3a1eaca8dfdde33",
  "protocol_sha256": "ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a",
  "historical_metadata_path": "docs/history/actions-artifacts-metadata.json",
  "historical_archive_directory": "experiments/archive/original-zips",
  "phase5": {
    "status": "completed",
    "run_id": 0,
    "run_attempt": 1,
    "artifact_id": 0,
    "archive_path": "experiments/closure/inputs/phase5-original.zip",
    "sha256": "SHA256_OFICIAL_DO_ZIP_DA_FASE5"
  },
  "analysis_directory": "experiments/closure/inputs/chat4-analysis",
  "evidence_directory": "experiments/closure/inputs/evidence",
  "limitations": []
}
```

O construtor verifica os seis digests históricos fixos, os manifests internos,
identidade da Fase 5, manifesto canônico aprovado, 32.200 chaves únicas da matriz,
limites e soma de passos, tentativa 1, ausência de treino e referências frozen.
Ele não reavalia os modelos. Arquivos de análise/evidência são escolhidos
explicitamente; caches, credenciais, URLs assinadas e pastas de sessão não
devem ser incluídos. Não há coleta automática de todo o diretório do usuário.

A construção cria um único ZIP, um recibo externo e uma recuperação local de
verificação. Em falha pode restar uma saída parcial para inspeção; não a promova
a ATUAL. O script exige campanha completa e não produz silenciosamente um
“pacote completo” quando falta o relatório ou a execução. Um fechamento com
impedimento deve ser registrado como tal, preservando as cópias anteriores.

## Recuperação e reutilização

1. Confirme o SHA-256 externo usando o registro independente de entrega. Em
   PowerShell, `Get-FileHash -Algorithm SHA256` calcula o digest do arquivo.
2. Extraia a cópia autenticada em uma pasta privada nova. Os membros do pacote
   não contêm links, caminhos absolutos ou travessia de diretórios.
3. Na pasta `recovery-tools/`, verifique o ZIP original novamente:

   ```text
   python scripts/build_current_package.py --verify ../NOVA-Neural-Arena-ATUAL.zip --expected-sha256 SHA256_EXTERNO_CONFIRMADO
   ```

   Se o ZIP estiver em outra pasta, use seu caminho real. A opção `--extract-to`
   pode realizar uma recuperação verificada em diretório ainda inexistente.
4. Restaure a fonte em uma pasta nova a partir da raiz extraída:

   ```text
   git clone source/full-source-history.bundle restored-source
   git -C restored-source switch -c restored-current SHA_FINAL_DO_INVENTORY
   ```

5. Para reproduzir uma fase histórica, use seu commit congelado e o freeze
   correspondente, em ambiente isolado. A restauração de pesos não preserva
   replay/RNG suficientes para retomar o treinamento antigo bit a bit.

Upload e download de retorno devem preservar tamanho e SHA-256 exatos. O recibo
local marca a verificação remota como pendente até que ela seja comprovada.
Somente depois da conferência do pacote novo em cada destino devem ir à lixeira
as cópias antigas cuja cobertura foi verificada por inventário. Não esvaziar
lixeiras, apagar a única cópia recuperável ou remover arquivos de outros projetos.
Se TeraBox ou outro destino estiver inacessível, registrar a operação pendente,
sem afirmar sincronização e sem eliminar a cópia válida anterior.
