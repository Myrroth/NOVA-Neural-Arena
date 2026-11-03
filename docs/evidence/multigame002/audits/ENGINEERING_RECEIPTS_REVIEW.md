# Revisão independente de recibos técnicos — campanha 002

Revisor: `engineering_receipts`. Somente leitura da fonte A, CSVs autenticados, metadados, logs originais e testes existentes. Nenhum jogo, treino, avaliação, importação de módulo científico ou atualização foi executado nesta revisão.

Fonte A: `dc71247b2269d41c8d2aeb8c7d489977fac3ad3a`. Protocolo: `93591ede5951438f92432fc6601773aaafe28b2ba5840309aa7bd3611170a324`. Execução científica 41551291622, tentativa 1. O JSON registra hashes, contagens e verificações.

|Critério|Parecer técnico|Limite|
|---|---|---|
|6 — política compartilhada|Sustentado por fonte e execução: CNN/GRU64/Q10 única por réplica, sem cabeças específicas; mesmo fingerprint nos 5 jogos por slot compartilhado.|3 réplicas independentes. Compartilhar pesos não prova transferência nem domínio dos 5 jogos.|
|7 — observação|RGB 32×32, ações/recompensas próprias e memória; máscara constante de controles; telemetria privada fora da rede. 39.688 pares com máscara constante por jogo.|Máscara pode identificar indiretamente o jogo. Auditoria numérica independente analisa efeitos de pixels/GRU; não há prova de semântica/objetos ou jogos arbitrários.|
|9 — persistência|Teste existente cobre 25 transições mecânicas e 25 atualizações só de replay sintético, perda até 1e-6, hash de pesos e estados físicos exatos. Logs originais A/B confirmam suíte completa: 474 testes e 31 subtestes passaram. Log de recuperação Linux 41619297256 informa 9 restaurados, VERIFIED, 0 interações e 0 atualizações.|Não é continuação científica dos modelos oficiais. Detalhes da recuperação estão no artefato separado a autenticar pelo root; esta revisão leu o log. A falha numérica do Windows deve ser preservada.|
|10 — holdout/orçamento|Racing ausente do pré-treino em CSV/metadados; 60.000 passos por jogo-fonte/réplica e por adaptação/scratch no alvo; sementes de treino ≥1.000.000, finais 920001..920020 disjuntas; sementes do alvo iguais por índice de episódio.|Durações/contagens de episódios diferem. Transferência e retenção medidas falharam; recibo técnico não muda FAIL.|
|11 — separação/limites|1.735 episódios/174.664 passos de avaliação; 1.080.000 de treino; 34 verificações independentes de limites passam.|Execução concluída não comprova a visão científica inteira. Preservar histórico 001.|

O analisador congelado valida apenas `status:PASS`, caminhos e hashes dos recibos. O responsável precisa revisar conteúdo e anexar os arquivos sob `evidence_root`; recibos não podem converter falhas numéricas em aprovação. A visão completa permanece `NOT_CONCLUDED`, pois 2–5, 8 e 10 falharam. Entrega/coordenação 12/13 estão fora deste parecer.
