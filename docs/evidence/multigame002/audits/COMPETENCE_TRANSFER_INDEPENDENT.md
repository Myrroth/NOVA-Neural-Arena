# Auditoria independente de competencia e transferencia

Fonte cientifica A: `dc71247b2269d41c8d2aeb8c7d489977fac3ad3a`. Protocolo SHA-256: `93591ede5951438f92432fc6601773aaafe28b2ba5840309aa7bd3611170a324`.

Recalculo direto de CSVs autenticados por SHA-256 e manifesto. Nao foram importados modulos da arena, chamado o analisador A, executados jogos, gradientes, novas sementes de jogo ou consultas de rede. O protocolo foi reconstruido estaticamente de A; o analisador congelado foi lido apenas como especificacao e seu resultado foi consultado depois do calculo independente.

Resultado restrito: **criterio 1 PASS; criterios 2, 3, 4, 5 e 10 FAIL**. Os 21 contrastes recalculados coincidem numericamente com o analisador congelado a tolerancia de 1e-12. Nenhuma conclusao de aceite geral ou do criterio 13 e emitida.

## Metodo e dados

- Matriz completa: 1735 celulas unicas, 174664 passos, episodios de ate 400 passos; sem celulas extras, removidas ou duplicadas.
- Tres replicas ajustadas independentes (1103, 2207, 3301), cada uma com 20 ambientes pareados (920001..920020). O controle random de seed0 e compartilhado entre replicas conforme o pre-registro.
- Bootstrap hierarquico proprio: 10.000 reamostragens; modelos primeiro, ambientes pareados depois; IC percentil de 95%. Seed-base510002 com o mesmo salt SHA-256 por contraste da fonte A. Interceptacao e colisoes/distancia usam somas dos numeradores e denominadores, sem media de taxas por episodio.
- PASS numerico exige ganho medio >= margem, limite inferior do IC >0 e pelo menos2/3 replicas com ganho positivo. A margem e absoluta contra random; contra untrained exige ganho positivo com IC positivo. Os tres modelos limitam a generalizacao dos ICs.

## Comparacoes contra random

| Jogo / medida | Compartilhado | Random | Ganho | Margem | IC95 do ganho | Replicas favoraveis | Resultado |
|---|---:|---:|---:|---:|---|---:|---|
| Snake / comidas | 2.8167 | 0.0000 | 2.8167 | 1.0000 | [1.9333; 3.9833] | 3/3 | PASS |
| Pong / pontos liquidos | -2.1500 | -3.0000 | 0.8500 | 0.5000 | [0.5333; 1.2833] | 3/3 | PASS |
| Pong / interceptacao | 0.3508 | 0.1549 | 0.1959 | 0.1000 | [0.0993; 0.2924] | 3/3 | PASS |
| Agar / comidas | 1.4667 | 1.1000 | 0.3667 | 2.0000 | [-0.1667; 0.9000] | 2/3 | FAIL |
| Agar / presas | 0.2667 | 0.0500 | 0.2167 | 0.3000 | [0.0500; 0.4000] | 3/3 | FAIL |
| Agar / sobrevivencia | 0.0000 | 0.0000 | 0.0000 | 0.1000 | [0.0000; 0.0000] | 0/3 | FAIL |
| Flappy / canos | 0.7333 | 0.0000 | 0.7333 | 1.0000 | [0.2500; 1.4667] | 3/3 | FAIL |
| Flappy / canos estreitos | 0.1500 | 0.0000 | 0.1500 | 0.0000 | [0.0167; 0.3667] | 3/3 | PASS |
| Flappy / canos largos | 0.5833 | 0.0000 | 0.5833 | 0.0000 | [0.2000; 1.1500] | 3/3 | PASS |
| Corrida / progresso [0,1] | 0.3238 | 0.1580 | 0.1658 | 0.1500 | [-0.0863; 0.4760] | 2/3 | FAIL |
| Corrida / voltas completas | 0.0833 | 0.0000 | 0.0833 | 0.1000 | [0.0000; 0.2500] | 1/3 | FAIL |

## Comparacoes contra rede nao treinada

| Jogo / medida | Nao treinada | Ganho | IC95 do ganho | Replicas favoraveis | Resultado |
|---|---:|---:|---|---:|---|
| Snake / comidas | 0.0000 | 2.8167 | [1.9333; 3.9833] | 3/3 | PASS |
| Pong / pontos liquidos | -2.4500 | 0.3000 | [-0.2337; 0.8833] | 2/3 | FAIL |
| Pong / interceptacao | 0.1844 | 0.1664 | [0.0742; 0.2687] | 3/3 | PASS |
| Agar / comidas | 1.1667 | 0.3000 | [-0.5500; 1.0333] | 2/3 | FAIL |
| Agar / presas | 0.2000 | 0.0667 | [-0.1333; 0.2333] | 2/3 | FAIL |
| Agar / sobrevivencia | 0.0000 | 0.0000 | [0.0000; 0.0000] | 0/3 | FAIL |
| Flappy / canos | 0.0000 | 0.7333 | [0.2333; 1.4833] | 3/3 | PASS |
| Corrida / progresso [0,1] | 0.1791 | 0.1447 | [-0.0338; 0.3747] | 2/3 | FAIL |
| Corrida / voltas completas | 0.0000 | 0.0833 | [0.0000; 0.2500] | 1/3 | FAIL |

## O que os resultados sustentam

- **Snake:** 2,8167 comidas/episodio apos adaptacao, frente a zero nos dois controles; IC95 do ganho [1,9333;3,9833],3/3 replicas positivas. Houve comidas em interior, parede e canto: medias3,2083/2,4815/2,7778 em24/27/9 episodios. O criterio numerico e a cobertura de spawn passam; a retencao posterior de80% falha separadamente.
- **Pong:** ha melhora de0,85 ponto liquido e19,59 pontos percentuais de interceptacao contra random; ambos passam suas margens. A interceptacao tambem supera untrained com IC positivo. O criterio2 falha porque o ganho de0,30 ponto liquido sobre untrained tem IC[-0,2337;0,8833], apesar de2/3 replicas positivas. A media final de pontos liquidos ainda e negativa(-2,15). O escopo permanece somente o oponente local assimetrico congelado.
- **Agar:** medias de1,4667 comidas e0,2667 presas; os ganhos contra random sao+0,3667 e+0,2167. O ganho em presas tem IC positivo e3/3 replicas favoraveis, mas fica abaixo da margem0,3. Comidas nao atingem margem2 e o IC cruza zero. Todos os60 episodios finais compartilhados terminam com eaten=1, assim como ambos os controles: nenhuma reducao de morte foi demonstrada. Exposicao a celulas grandes e pequenas foi registrada em cada replica; isso nao prova fuga ou perseguicao causal. A auditoria de cenarios continua NOT_EVIDENCED nesta revisao de CSV.
- **Flappy:**0,7333 cano/episodio, contra zero em random/untrained, com IC positivo e3/3 replicas favoraveis. Os ganhos fisicos de canos estreitos e largos tambem possuem IC positivo. O criterio4 falha somente pela margem absoluta de+1 cano contra random; este FAIL preserva a evidencia de aprendizado parcial.
- **Corrida compartilhada:** progresso medio32,38% contra15,80%random e17,91%untrained. O ganho medio contra random ultrapassa a margem15pp, mas seu IC[-8,63pp;47,60pp] cruza zero; contra untrained tambem cruza zero. Voltas completas8,33% contra zero nos controles ficam abaixo da margem10pp, apenas1/3 replica e favoravel e o IC toca zero. Colisoes por distancia e o recibo de curvas/aceleracao/freio sao componentes separados; seus valores e limites estao no JSON.

## Transferencia e AUC do alvo

O compartilhado adaptado e inferior ao treinamento do zero nesta campanha. Progresso final:32,38% contra88,37% from_scratch, ganho **-55,99pp**, IC95[-86,63pp;-29,90pp],0/3 replicas favoraveis. From_scratch completa75% das voltas; o compartilhado completa8,33%. Isso demonstra habilidade adquirida no alvo pelo controle scratch, sem demonstrar transferencia positiva do compartilhado.

| Replica | AUC adaptacao | AUC scratch | Ganho relativo | Passos por ramo |
|---|---:|---:|---:|---:|
| 1103 | 0.429175 | 0.496239 | -13.51% | 60.000 |
| 2207 | 0.148679 | 0.509936 | -70.84% | 60.000 |
| 3301 | 0.271627 | 0.491513 | -44.74% | 60.000 |

AUC: soma do progresso normalizado no extremo direito de cada episodio/episodio parcial, ponderada pelos passos desde o extremo anterior, dividida por60.000. Os seis ramos terminam exatamente em60.000 e a soma de passos dos episodios coincide com a cobertura. Ganho relativo medio **-43,03%**, contra margem+10%;0/3 replicas positivas. E uma medida da politica exploratoria durante o treino, nao de uma politica final deterministica.

## Retencao das fontes

Para cada replica e jogo, a razao e (ganho apos adaptacao sobre random)/(ganho antes sobre random). Todos os ganhos anteriores sao positivos; exige-se>=80% em cada uma das12 combinacoes.

| Jogo | Antes: ganhos1103/2207/3301 | Depois: ganhos1103/2207/3301 | Retencao1103/2207/3301 | Resultado |
|---|---|---|---|---|
| Snake | 4.550 / 6.000 / 5.100 | 2.350 / 3.200 / 2.900 | 51.6% / 53.3% / 56.9% | FAIL |
| Pong | 1.750 / 3.500 / 1.700 | 1.150 / 0.750 / 0.650 | 65.7% / 21.4% / 38.2% | FAIL |
| Agar | 1.200 / 0.700 / 0.650 | -0.100 / 0.600 / 0.600 | -8.3% / 85.7% / 92.3% | FAIL |
| Flappy | 0.900 / 0.250 / 1.050 | 0.300 / 0.400 / 1.500 | 33.3% / 160.0% / 142.9% | FAIL |

A retencao falha nos quatro jogos: Snake e Pong ficam abaixo de80% nas tres replicas; Agar falha em1103, mas2207/3301 preservam85,7%/92,3%; Flappy falha em1103, enquanto2207/3301 melhoram para160,0%/142,9%. Estas diferencas individuais nao sao ocultadas por uma media global.

## Fontes, autenticacao e limites

Artifact real `13195542205`, run `41551291622`, ZIP SHA-256 `f396acfcbc8de35e67cbe79be4b6b986d7b536019bca2a62a3effa60ccefa129`. ROOT_SHA256SUMS: `2797b5280d367be9e067d8528f2b0c8973081342eb248aacca19054475b18840`.

| Fonte local | Bytes | SHA-256 |
|---|---:|---|
| `DATA_PROVENANCE.json` | 1783 | `d1a4253d050947e9a15d84bb3c22accd21b7de0827b4b09119eaaa64852971c0` |
| `ROOT_SHA256SUMS` | 528030 | `2797b5280d367be9e067d8528f2b0c8973081342eb248aacca19054475b18840` |
| `episodes.csv` | 502563 | `034ad7329d3faa93452b022a2c38ccc801bd39ee2068adf0f911e83e0d0e1f08` |
| `training_curve.csv` | 5501973 | `2bc186e834cfa880159597179df792a5f4a80cf2e24c64a3ac20e9563d68e115` |
| `action_pairs.csv` | 20413795 | `a98841203bec14abff2633d93a8b6264f4213118e852d030bc1fc4bb01c1f1be` |
| `model_predictions.csv` | 12921710 | `b45a32ba039da29d83275c06be47b1975410bb5f39b8bd72dcebf9175ca75945` |
| `ANALYSIS_RAW.json` | 62110 | `d560e92f3e84d6e9b2bee15b8a192d6a2adb5984e53b753b764eed15a5c65b74` |

Cada CSV foi comparado diretamente ao manifesto autenticado e ao DATA_PROVENANCE. Checkpoints, replay e estrutura completa do ZIP pertencem a auditoria separada; esta revisao nao autenticou seus conteudos. Recibos de cenarios de Agar, controle de corrida e holdout/orcamento interno continuam NOT_EVIDENCED neste escopo. Os valores completos por modelo, condicoes individuais, rotulos de bootstrap e hashes das fontes A estao no JSON acompanhante.

Os limiares e todas as sementes foram preservados. Falhas e ganhos parciais estao registrados sem retreino, selecao de sementes, novos episodios ou alteracao de metodo.
