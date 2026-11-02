# Verificação independente do fechamento NNA-FECHAMENTO-001

Auditoria em 02/11/2026, sobre a base técnica `8468e0dad3a75ecfe06187d8dbbf7a523f468df0`. Esta auditoria não executou a campanha científica, não carregou checkpoints e não treinou modelos.

## Código histórico e executor

Comparações Git locais confirmaram ausência de alterações nos sete executáveis históricos: Snake, DQN, treinador, benchmark e avaliadores das Fases 2–4, contra os respectivos commits congelados. Do executor `39461168db30edb90a25385fb3a1eaca8dfdde33` ao HEAD aprovado `8468e0d`, o diff contém somente os dois workflows da Fase 5 e os dois arquivos de testes; o código científico permanece igual.

O workflow `nna-phase5-once.yml` fixa literalmente esse executor e o hash do protocolo. Exige evento de etiqueta específica na PR #8, branches e repositórios corretos; recusa tentativa maior que 1; serializa execuções num grupo global; procura uma reserva persistente na Issue #6 antes de criar uma única reserva e antes do primeiro episódio. Uma reserva já existente impede repetição, inclusive após falha. As Actions usam SHAs completos. Permissões são leitura de código/Actions e escrita de Issues somente para registrar a reserva.

O limite declarado é CPU, 60 minutos, 32.200 episódios, 200 passos por episódio e 6.440.000 passos totais. O executor utiliza redes congeladas, compara fingerprints antes/depois, não cria otimizador, não salva modelos e grava saídas novas. Dados parciais são preservados quando possível; uma falha não autoriza rerun. A auditoria do workflow não é uma afirmação de que a campanha terminou: run, artefato, integridade e resultado científico devem ser verificados separadamente.

## Geometria e protocolo

A avaliação independente somente das definições puras presentes no código, sem importar RL ou executar jogos, reproduziu:

- 24 spawns válidos de três segmentos;
- 92 condições, sendo 48 de alimento físico e 44 relativo;
- sete políticas e 50 seeds ambientais;
- 32.200 combinações de avaliação e orçamento máximo de 6.440.000 passos;
- seis contrastes primários, intervalos descritivos e ausência de testes confirmatórios/equivalência.

Hash canônico recomputado: `ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a`, exatamente igual à [decisão formal do Planejamento](https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-6568072221). A enumeração não executou episódios.

## Evidência de CI e Pong

O log da [CI integrada 41381116539](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41381116539), job `125071939838`, registra **187 testes aprovados em 4,53 s**, com dois avisos informativos do verificador de renderização Gymnasium sem `spec`. O comando foi `python -m pytest -q` para toda a suíte, incluindo os testes de Pong, multijogo, relatórios e executor. O log também imprimiu o mesmo hash canônico do protocolo. A [CI específica do Pong 41380607982](https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/41380607982) é uma evidência adicional da PR #7, sem treinamento.

Uma auditoria de alcançabilidade identificou uma limitação concreta no Pong original da PR #7: no tabuleiro padrão 20×12, o oponente direito acompanha a bola com velocidade suficiente para nunca perder. As quatro velocidades possíveis de saque entram em um ciclo determinístico de 352 transições, sem pontos, mesmo concedendo interceptação perfeita à esquerda. Como rebater na esquerda não altera a velocidade vertical, qualquer erro da esquerda apenas dá ponto à direita e retorna a um dos mesmos saques. Assim, recompensa positiva da esquerda é inalcançável no modo padrão original. O teste unitário que força manualmente uma posição perdida à direita verifica a regra de pontuação, mas não demonstra alcançabilidade por ações legais a partir do reset.

Esse achado não afeta o Snake nem a campanha congelada da Fase 5. A árvore de fechamento contém a correção isolada `pong-local-v2`: o adversário direito reage a cada dois ticks, em vez de acompanhar a bola a cada tick. A recompensa permanece `pong-points-v1`; a identidade do ambiente muda porque a dinâmica mudou, e o contrato recusa checkpoints identificados como v1. A fonte original e o executor aprovado continuam preservados nos commits históricos, que não foram reescritos.

Foram acrescentados testes que verificam cadência/ausência de sorteios extras e alcançam +1 a partir de `reset(seed)`, sem alterar o estado, usando somente observações públicas e ações legais de uma fixture explicitamente declarada. A verificação finita dos métodos reais da nova versão encontrou pontuação positiva em 25 ou 40 passos para cada uma das quatro velocidades iniciais possíveis; isso prova alcançabilidade do objetivo, não aprendizado. Compilação passou localmente. A evidência dos testes completos de Gymnasium/Pong v2 deve vir da CI da PR de fechamento, separadamente da CI187 da versão anterior; este registro não a presume executada.

## Limites de interpretação

Há três treinamentos independentes, e não 50 ou 32.200. Seeds ambientais repetidas não tornam os cenários independentes nem garantem alimentos futuros fisicamente iguais. Comida inicial fixa e políticas determinísticas podem produzir trajetórias repetidas antes da primeira coleta. Os intervalos planejados são condicionais aos checkpoints e cenários, com reamostragem de blocos de seed nas agregações; não demonstram equivalência, não inferioridade ou transferência de habilidades para Pong.

O resultado científico definitivo, os arquivos ATUAL e a confirmação de substituição das cópias redundantes no Drive/TeraBox pertencem ao aceite final da missão. Este documento é uma evidência técnica de preparação e revisão, não substitui essas verificações.
