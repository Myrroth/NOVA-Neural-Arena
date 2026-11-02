# Fase 5 — condições difíceis (proposta v1, NÃO AUTORIZADA)

Este documento pré-registra uma **proposta** de desenho e a infraestrutura a ser
revisada pelo Chat 1. Nenhuma campanha da Fase 5 foi executada. A linha de comando
somente enumera e valida; `--campaign` é recusado e não existe `--authorized`.
Será necessária uma revisão futura com autorização expressa, commit imutável do
avaliador, hash do protocolo e workflow científico próprio. Um argumento de
linha de comando ou etiqueta isolada não constitui autorização.

## Evidência e pergunta

O [Relatório Científico 005](https://github.com/Myrroth/NOVA-Neural-Arena/pull/4#issuecomment-6567175661)
encontrou robustez restrita a 12 estados relativamente centrais no Snake 10×10.
Não demonstrou equivalência, invariância universal, aprendizagem nova ou
transferência para outros jogos. A Fase 5 proposta investiga limites próximos a
paredes/cantos, mantendo pesos, regras, observação 11D e recompensas originais.

Hipóteses candidatas:

1. DQNs treinadas coletam mais alimentos que a rede não treinada correspondente
   e a política aleatória, dentro de cada condição inicial difícil.
2. Estados de parede/canto reduzem score e probabilidade de primeiro alimento em
   relação ao centro com direção e controle de comida correspondentes.
3. Longa sobrevivência sem alimento pode vir acompanhada de estados repetidos;
   sobreviver até 200 passos não equivale a competência ou vitória.

Não há hipótese de não inferioridade: a margem em alimentos e o poder necessário
dependem de decisão científica do Chat 1, antes de qualquer execução oficial.

## Matriz exata candidata

- Snake 10×10; corpo inicial com três segmentos alinhados atrás da cabeça.
- Cabeças propostas: `(5,5)`, `(0,5)`, `(9,5)`, `(5,0)`, `(5,9)`, `(0,0)`,
  `(0,9)`, `(9,0)`, `(9,9)`; direções leste, sul, oeste, norte.
- Cada corpo é validado antes de incluir a condição. Não se desloca nem se
  encurta um corpo inválido. Das 36 combinações cabeça/direção, 12 são excluídas
  por corpo fora dos limites; restam **24 spawns**: 4 centrais, 12 de parede,
  8 de canto. Orientação voltada para fora pode ser válida e causar colisão no
  primeiro movimento em frente; isso é uma condição de risco deliberada.
- Dois controles **físicos absolutos**: primeira comida em `(4,4)` ou `(6,6)`.
  Ambos são livres em todos os 24 spawns: **48 condições**. Compartilham a
  posição física, mas não a distância relativa à cabeça ou às paredes.
- Três controles **relativos à orientação**: `(frente,direita)` em `(1,0)`,
  `(0,1)`, `(0,-1)`. São excluídos deslocamentos fora do tabuleiro ou sobre o
  corpo: 28 de 72 candidatos excluídos; **44 condições** válidas. Preservam
  distância/deslocamento relativo, mas não a posição física absoluta.
- Total: **92 condições**, identificadores estáveis contendo cabeça, direção,
  modo e coordenadas de comida. A enumeração completa é retornada por
  `python -m arena.evaluation.phase5_hardstates --validate-only`, junto ao hash
  SHA-256 determinístico do protocolo. O comando não reseta nem joga um episódio.
- Sete políticas: checkpoints oficiais treinados 11/42/73; três redes de mesma
  arquitetura inicializadas com seeds 11/42/73 e não treinadas; uma aleatória.
- Seeds ambientais propostas: **500001–500050**, inclusive, sem treino.
  Política aleatória: `random.Random(90000000 + environmental_seed)`, reiniciada
  por condição. Redes usam argmax guloso sob `torch.inference_mode()`.
- Limite: **200 passos** por episódio, regras originais de término/truncamento.
- Proposta completa: **92 × 50 × 7 = 32.200 episódios**; orçamento máximo de
  6.440.000 passos. Essa contagem é somente enumerada. Chat 1 pode revisar o
  tamanho amostral e a matriz, exigindo nova versão/hash antes da aprovação.

## Comida e números aleatórios

`HardStateSnakeEnv` chama apenas o reset Gymnasium para inicializar o gerador,
monta o corpo validado e atribui a primeira comida explicitamente. **Zero números
aleatórios são consumidos no reset**, independentemente do modo ou condição.
Não há spawn descartado, amostragem por rejeição nem tentativa de corrigir
geometria. As sete políticas recebem a mesma comida dentro da condição.

Depois do primeiro alimento, o método original `_spawn_food` escolhe uma célula
livre em ordem por linhas. A mesma seed mantém a mesma sequência de números
aleatórios por evento de spawn; corpos/trajetórias distintos podem mapear os
mesmos números para posições diferentes. Portanto não se afirma igualdade das
trajetórias ou de todos os alimentos futuros entre estados. A comida inicial é
constante dentro de uma condição, e as 50 seeds afetam os spawns posteriores;
episódios sem alcançar comida podem ser idênticos entre seeds para uma DQN
determinística. Isso reduz informação efetiva e deverá aparecer na análise.

## Pesos, integridade e proveniência

Os três checkpoints têm origem no treino congelado
`24bd38e9e78a14e894f32d7d1c5b5b6d6d724463`:

| Seed | Artifact do piloto | SHA-256 de model.pt |
|---|---:|---|
| 11 | 13101471957 | ff132bc23aaf5b56d00679a07961d23c2de30bafa7bbaa4e99ff966b61a49130 |
| 42 | 13101072100 | 0623256abf3b068f212bacaa5c46f8bbcc6aabf2aeaa0f7ef5ecdc0ec8120cbe |
| 73 | 13100988601 | 2899df84afe4512ca127bf3367490a5bc42c88f3ad7868daf8195ed419ecb8fe |

`verify_official_policies` verifica manifesto, metadados e hashes, usa o novo
loader seguro e congela as seis redes (`eval`, sem gradientes), **sem jogar**.
`FrozenPolicies.assert_frozen` compara fingerprints e estado de gradientes antes
e depois de uso. O executor científico oficial permanece desabilitado. Nenhum
checkpoint original é salvo, reserializado ou substituído.

O gravador de resultados exige diretório inexistente e arquivos novos; uma
reexecução no mesmo destino falha. A validação sintética registra versões,
configuração, cenários, fingerprints, hashes do código e SHA256SUMS. A futura
campanha deverá acrescentar commit exato, protocolo aprovado, prova/link da
autorização, run/attempt e manifesto das cópias duráveis de origem.

## Desfechos e limites de interpretação

Primário: **score (número de alimentos)**, condicionado a cada checkpoint.
Secundários: recompensa acumulada, passos, motivo de término, alimento por 100
passos, primeiro passo com alimento, observação censurada quando não alcançado,
maior intervalo até/entre alimentos (incluindo cauda final), estados repetidos.

Estado repetido significa igualdade da **sequência completa do corpo, direção
e comida**, não da observação compacta 11D. Registram-se primeiro passo de
repetição, período observado e número de visitas repetidas. Colisão terminal
não é contada como repetição do corpo parado. A instrumentação **não termina
episódios antecipadamente**. Repetição é um candidato a ciclo; para uma política
aleatória não prova comportamento periódico. RNG do ambiente/política e limite
de passos não integram a chave; não se afirma recorrência de todo o simulador.

`first_food_step` nulo significa censura no horizonte observado, que pode ser
colisão precoce. A média somente entre sucessos é sujeita a seleção e não deve
ser comparada sem informar também a taxa de obtenção e censura. O nome
`max_steps_between_food_events` inclui o passo que completa o intervalo com um
alimento; não significa exclusivamente passos com recompensa negativa.

A recompensa original permanece `+1` por alimento, `-1` por colisão e `-0,01`
por passo comum. Longos caminhos podem ser penalizados mesmo quando corretos;
uma trajetória sem comida pode preferir colidir cedo em termos de retorno não
descontado. Por exemplo, 200 passos sem comida retornam `-2`, e uma colisão no
primeiro passo retorna `-1`. Isso não prova falha do aprendizado observado.
O objetivo DQN usa `gamma=0,98`: os mesmos 200 passos comuns têm retorno
descontado de aproximadamente `-0,491`, superior a `-1` da colisão imediata.
Não se pode inferir incentivo à colisão somente pela soma registrada no CSV.
Alterar a recompensa exigiria outro experimento e novos modelos independentes.

## Pareamento, incerteza e multiplicidade

Cada comparação deve usar a chave de seed, nunca a ordem das linhas. O utilitário
`paired_score_interval` rejeita seeds duplicadas, ausentes, não inteiras e scores
não finitos, e usa bootstrap pareado percentil de 95%, **10.000 reamostragens**,
seed **510000**. Os intervalos são descritivos e condicionais aos pesos fixos;
não constituem automaticamente testes ou intervalos simultâneos.

Famílias candidatas, a fixar antes de executar:

1. **552 contrastes** de treinada versus não treinada correspondente/aleatória:
   92 condições × 3 checkpoints × 2 controles, com mesmas seeds e comida.
2. **216 contrastes** de condição difícil versus centro, por checkpoint:
   (40 controles físicos não centrais + 32 relativos não centrais) × 3.
   A referência central mantém a mesma direção e o mesmo modo/chave de comida.
   Comparações físicas e relativas respondem perguntas distintas; não são
   tratadas como um efeito causal isolado da orientação.

`holm_adjust` corrige uma família completa de p-valores fornecidos por um teste
válido, previamente declarado. Não calcula p-valores a partir de intervalos de
bootstrap. A escolha/implementação final do teste, unilateralidade, poder,
controle de dependência e famílias primárias são **pendências do Chat 1**. Até
essa revisão, não se faz conclusão de significância, equivalência ou não
inferioridade. Três modelos não são 50×92 treinos independentes; métricas e
contrastes por cenário são correlacionados.

## Validação técnica disponível

`tests/test_phase5_hardstates.py` verifica geometria, exclusões, equivalência
física/relativa, zero draws de RNG, primeira comida, censura, colisão, repetição
de estado completo sem parada antecipada, saídas sem sobrescrita, bootstrap,
Holm e bloqueio da CLI. O microteste `synthetic_smoke` executa somente **14
episódios de até 12 passos**, com redes recém-inicializadas e política aleatória,
sem arquivos de checkpoints oficiais ou treino. Saídas são marcadas
`SYNTHETIC_TECHNICAL_VALIDATION`; não são resultados científicos da Fase 5.

## Decisões necessárias antes de uma campanha oficial

Chat 1 deve aprovar a matriz, número de seeds/poder, contraste primário, testes
e multiplicidade, eventual margem de não inferioridade, orçamento e executor
imutável. Em seguida, uma revisão implementará um workflow de campanha separado
com commit e hash literais, auditoria dos arquivos históricos, autorização
específica e prevenção de repetição automática. Nesta entrega o workflow da
Fase 5 é somente de infraestrutura; nenhuma etiqueta inicia campanha.
