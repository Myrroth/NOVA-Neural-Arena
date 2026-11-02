# Estado consolidado — NOVA Neural Arena

**Registro histórico de 01/11/2026.** O estado vigente está no
[fechamento de 02/11/2026](CLOSURE_STATUS.md): a avaliação única da Fase 5
terminou com 32.200 episódios e 3.212.125 passos, sem treinamento, e a CI
final aprovou 203 testes e 31 subtestes. Os bloqueios de preparação abaixo
foram superados pelas decisões e evidências desse fechamento; não são uma
ordem de repetir campanhas. Pong v2 e recuperação do ATUAL estão documentados.

Atualizado em 01/11/2026, America/Sao_Paulo. Este documento consolida o estado atual; entradas antigas de `STATUS.md`, corpos das PRs e o roadmap inicial continuam como registros das decisões em seus respectivos momentos. A NOVA Neural Arena é independente da IA assistente NOVA original.

## Resultado histórico confirmado

As quatro campanhas anteriores estão concluídas operacionalmente, todas com `success` e tentativa 1 no GitHub. As PRs #1–#4 permanecem abertas em rascunho, sem merge. Nenhuma campanha histórica precisa ser repetida para esta consolidação.

| Fase científica | Run oficial | Evidência e limite |
|---|---|---|
| 1: piloto Snake/DQN | [41350959549](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41350959549) | Três modelos treinados, seeds 11/42/73; desempenho superior à aleatória em Snake 10×10. Sem controle neural inicial no piloto. |
| 2: redes não treinadas | [41367188257](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41367188257) | 350 avaliações congeladas; evidência de aprendizado dentro da distribuição. |
| 3: dimensões 8/10/12 | [41370622154](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41370622154) | 1.050 avaliações; robustez limitada à mudança de dimensões no mesmo Snake. |
| 4: posição/orientação | [41373651632](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41373651632) | 4.200 avaliações; robustez limitada aos 12 estados iniciais testados. |

Há somente **três treinamentos independentes**. As milhares de avaliações não são novos treinamentos, nem demonstram transferência para outros jogos ou equivalência universal entre estados.

O [Relatório Científico nº 005](https://github.com/Myrroth/NOVA-Neural-Arena/pull/4#issuecomment-6567175661) é a referência da Fase 4. A [auditoria histórica](HISTORICAL_AUDIT.md) reúne os relatórios disponíveis, hashes, commits, metadados e ressalvas. As denominações literais dos relatórios 001/003 não foram confirmadas; os pareceres correspondentes foram preservados sem renumeração silenciosa.

## Preservação e recuperação

- Catálogo oficial: seis ZIPs, com digests/tamanhos/expirações em [history/actions-artifacts-metadata.json](history/actions-artifacts-metadata.json).
- Conversas científicas e decisões das quatro PRs: [history/prs-1-4-snapshot.json](history/prs-1-4-snapshot.json).
- Recuperação offline: `arena.shared.preservation` valida os seis ZIPs contra identidades fixas, confere os manifestos e contagens e extrai cópias novas; não substitui checkpoints existentes.
- Os ZIPs originais devem conservar seus bytes e nomes. Metadados adicionais pertencem ao inventário externo, especialmente na Fase 2, cujo manifesto histórico cobre somente três arquivos científicos.
- Os artefatos do Actions expiram em **01/12/2026 no horário de São Paulo**, entre 18:14:59 e 22:40:58. Nesta entrega, os seis ZIPs tiveram hashes externos conferidos, 38 hashes de manifesto interno foram aprovados e três arquivos adicionais da Fase 2 foram ancorados pelo hash do ZIP original. Um bundle Git completo com sete refs também passou por verificação.
- O pacote privado de preservação no Google Drive foi relido e conferido: **455.899 bytes**, SHA-256 `692f7e1258741a431699f6e7d87c47ab7666a08a94dabeec845dd3a815caed19`; 17 hashes de conteúdo do pacote aprovados. As localizações e o procedimento de recuperação constam de [PRESERVATION.md](PRESERVATION.md). Essa verificação de cópia não equivale à execução da suíte de testes ou da campanha.

## Infraestrutura adicionada para revisão

| Componente | Função |
|---|---|
| `arena.shared.integrity` | Caminhos confinados, verificação de manifestos, leitura limitada e identidade SHA-256. |
| `arena.shared.checkpoints` | Carregamento de bytes verificados em CPU, `weights_only=True`, configuração/tensores validados e rede congelada; não cria otimizador nem executa partida. |
| `arena.shared.preservation` | Recuperação dos seis pacotes oficiais com verificação externa/interna, sem sobrescrita. |
| `arena.evaluation.phase5_hardstates` | Cenários geométricos de bordas/cantos, comida controlada, métricas e comparação pareada; CLI somente para validação do desenho. |
| `arena.evaluation.reporting` | Relatórios JSON/Markdown e gráficos PNG/SVG opcionais, a partir de CSVs existentes verificados, sem partidas ou modelos. |
| `.github/workflows/lab-infrastructure.yml` | CI da PR baseada na Fase 4: preservação do código histórico, dependências oficiais e testes pequenos/sintéticos. |

O CI novo não baixa artefatos temporários do Actions; assim sua manutenção não depende da retenção científica. A auditoria dos modelos oficiais deve usar as cópias preservadas e seus digests. Os módulos antigos do Snake, DQN, treinador, benchmark e avaliadores congelados continuam protegidos por comparação Git contra os SHAs históricos.

## Validação e seu significado

Os testes novos cobrem adulteração de arquivos, caminhos inseguros, configuração/tensores de checkpoints e limites de execução; os testes da Fase 5 usam fixtures sintéticas e matrizes pequenas. A suíte completa inclui microtestes legados de treinamento com modelos sintéticos temporários; ela não retreina os modelos oficiais.

Comandos reproduzíveis no ambiente oficial Python 3.11:

```bash
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-ci.txt -e '.[dev]'
python -m pip check
python -m compileall -q arena tests
python -m pytest -q
python -m arena.evaluation.phase5_hardstates --validate-only
```

Nesta preparação, o ambiente local não conseguiu obter as dependências por falhas de conectividade. Os três testes estáticos do workflow foram executados diretamente com Python 3.14 e aprovados, sem importar Gymnasium/PyTorch; a sintaxe do arquivo de testes também foi verificada. O resultado da suíte completa deve ser registrado pelo run de CI desta entrega; não confundir essas verificações estáticas com execução comprovada dos testes científicos. A validação da Fase 5 enumera e confere o desenho sem jogar a campanha proposta.

**Evidência executada:** a [CI inicial 41377237661](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41377237661), sobre a infraestrutura `6faec4b5a2524a801e71c7f72f8e4859a2eb5d2b`, aprovou **140 testes em 3,77 s**, `pip check`, compilação, comparação dos sete executáveis históricos e enumeração da Fase 5. Os dois avisos são informativos do Gymnasium sobre modos de renderização sem `spec`. Ambiente: Python 3.11.17, torch 2.10.0+cpu, NumPy 2.3.5, Gymnasium 1.4.0 e pytest 9.0.2. As evidências de commits posteriores e da auditoria opcional de carregamento estão na [PR #5](https://github.com/Myrroth/NOVA-Neural-Arena/pull/5).

O workflow separado `lab-checkpoint-audit.yml` é uma auditoria técnica opcional da PR #5, acionada somente pela etiqueta `lab-checkpoint-load-audit`. Ele fixa literalmente o carregador já aprovado na CI inicial, baixa os três artefatos originais, verifica e carrega pesos congelados sem inferência, partidas ou otimizador. Não inicia campanha. Sua dependência dos artefatos temporários não afeta a CI sintética; após a expiração, use as cópias duráveis e o mesmo carregador. As Actions dos dois workflows novos são fixadas por SHA completo, preservando os workflows históricos intactos.

A [segunda CI 41377667604](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41377667604) aprovou **142 testes em 2,34 s**, com os mesmos dois avisos informativos. A [auditoria de carregamento 41377668039](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41377668039), tentativa 1, também teve sucesso: três modelos oficiais verificados/carregados, parâmetros congelados e RNG preservado, **zero partidas e zero treinamento**. Seus resultados foram preservados em [history/checkpoint-load-audit.json](history/checkpoint-load-audit.json). A etiqueta técnica foi removida após a execução única.

O [gerador de relatórios](REPORTING.md) foi aplicado à cópia preservada da Fase 4, ancorada no hash oficial do CSV: **4.200 linhas e 84 grupos**. Os derivados novos em `experiments/reports/phase4-descriptive-v2/` são uma reanálise descritiva, separada dos resultados oficiais intactos. A CI também testa a exportação de gráficos com Matplotlib 3.10.7 e conserva uma prévia baseada exclusivamente em quatro linhas sintéticas de teste; ela não representa desempenho científico.

## Fase 5: preparação autorizada, campanha bloqueada

O desenho candidato inclui cabeças centrais, em paredes e em cantos; orientações são aceitas somente quando os três segmentos estão dentro do tabuleiro e não se sobrepõem. As primeiras comidas são fixadas por coordenadas físicas ou relativas válidas, sem consumir sorteios do RNG durante a inicialização. Após a primeira coleta, as trajetórias podem divergir; não afirmar que toda a sequência de alimentos seja fisicamente equivalente.

As políticas previstas continuam sendo os três checkpoints oficiais 11/42/73, três controles neurais não treinados e uma política aleatória. Score é o desfecho primário; primeira comida, eficiência, recompensa, colisões e repetição de estado são métricas complementares. Tempo até a primeira comida deve acompanhar a probabilidade de não obtê-la, evitando interpretar somente os casos bem-sucedidos.

O manifesto produzido por `--validate-only` é a fonte da matriz exata, seeds e hash do protocolo candidato. O desenho ainda não é uma autorização. `--campaign` é rejeitado nesta revisão; não existe workflow novo capaz de executar a campanha oficial por etiqueta ou acionamento manual.

Antes de qualquer campanha, o Chat 1 deve aprovar explicitamente: hipóteses, subconjunto/matriz final de cenários, seeds, quantidade de episódios e orçamento; endpoint primário e famílias de comparações; pareamento físico/relativo admissível; margem e poder caso haja teste de não inferioridade; e SHA literal do avaliador validado. O workflow executável deverá ser preparado em revisão posterior, com esse SHA e a autorização concreta. Nenhuma etiqueta histórica autoriza esta fase.

Próximos experimentos independentes e arquitetura para outros jogos estão no [roteiro de pesquisa](RESEARCH_ROADMAP.md). Não há merge, novo treinamento extenso, campanha da Fase 5 ou alteração da NOVA original autorizado por esta consolidação.
