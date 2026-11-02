# Roteiro técnico e científico

Propostas para revisão do Chat 1, em 01/11/2026. Este roteiro não inicia campanhas, não muda pesos históricos e não autoriza gastos. As Fases científicas 1–4 são as registradas nas PRs correspondentes; os nomes abaixo são etapas futuras, sem reutilizar a numeração antiga de Pong/Agar do contexto inicial.

## Primeiro: concluir a base verificável

Conservar os seis ZIPs originais, seus digests e metadados em armazenamento persistente privado, confirmar a leitura de volta e manter uma segunda cópia independente. Conservar também um bundle Git que contenha os commits executáveis congelados, para que a expiração do Actions não impeça restaurar o código. Registrar restaurações em diretórios novos, nunca sobre os originais. Uma identidade SHA confiável precisa vir do catálogo oficial, não do mesmo arquivo que se deseja validar.

O carregador novo usa bytes verificados, CPU e `weights_only=True`, valida configuração/tensores e retorna somente a rede em inferência. Isso reduz a superfície de desserialização, mas não transforma um checkpoint arbitrário em arquivo confiável: a documentação de [torch.load no PyTorch 2.10](https://docs.pytorch.org/docs/2.10/generated/torch.load.html) mantém a advertência contra origens não confiáveis. O digest conhecido e os limites de tamanho continuam necessários.

Preservar os avaliadores congelados e concentrar correções em componentes novos. A duplicação histórica não deve ser eliminada alterando o código que produziu os resultados; novas fases podem migrar para serviços compartilhados por meio de testes de paridade, deixando o legado disponível para recuperação.

## Arquitetura de evolução

| Camada | Contrato proposto | Estado desta entrega |
|---|---|---|
| Ambiente | `reset(seed)` / `step(action)` com observação, recompensa, `terminated`, `truncated` e metadados documentados. | Snake existente e adaptador de avaliação difícil; contratos multijogo ainda futuros. |
| Configuração e cenários | Config imutável, gerador puro, validação geométrica, manifesto canônico e hash do protocolo. | Implementado para o candidato da Fase 5. |
| Identidade e armazenamento | Conteúdo identificado por SHA, catálogo de proveniência, destinos novos e recuperação verificável. | Novos utilitários de integridade/preservação e catálogo histórico. |
| Política congelada | Carregar rede válida, agir sem gradientes, verificar fingerprint antes/depois, distinguir treinada/controle/aleatória. | Reutiliza QNetwork/observação Snake; não é política genérica para Pong. |
| Executor | Aplicar orçamento e seeds, separar RNG de política/ambiente, registrar um episódio e rejeitar saída existente. | Instrumentação técnica da Fase 5; campanha CLI desativada. |
| Análise | Validar matriz, parear chaves declaradas, sumarizar por condição e emitir intervalos com método explícito. | Ferramentas candidatas; inferência oficial depende do protocolo aprovado. |
| Autorização | Associar aprovação humana, protocolo e SHA executável; privilégios mínimos e trilha de auditoria. | CI sintética; workflow de campanha futura ainda não habilitado. |

Uma futura interface multijogo deve declarar dimensões e semântica da observação/ação, versão da recompensa, orçamento e critério de sucesso. A observação 11D do Snake e as três ações relativas não possuem significado automático em Pong. Compartilhar configuração, métricas, verificação e executor é uma forma imediata de reutilização; compartilhar pesos exige hipótese e experimento próprios.

## Etapa seguinte: limites da robustez no Snake

Revisar o manifesto candidato da Fase 5 antes de escolher a matriz oficial. Separar duas perguntas: efeito da orientação com a mesma comida física e efeito do cenário com comida relativa equivalente. Excluir e registrar geometrias inválidas, em vez de corrigir silenciosamente posição ou corpo. Quando uma comparação não possui equivalente válido, não preencher a célula com outro cenário.

Manter score como endpoint primário e apresentar colisão, tempo até a primeira comida, probabilidade de obter comida e eficiência por 100 passos separadamente. A média de `score/steps` por episódio e a razão `sum(score)/sum(steps)` respondem a perguntas diferentes; nomear a usada. Tratar repetição de estado como diagnóstico observável, com corpo inteiro, direção e comida na chave: uma cabeça repetida sozinha não caracteriza ciclo.

Usar seeds comuns para comparar políticas no mesmo cenário. Igualdade numérica da seed entre cenários não garante a mesma distribuição física de comida. Registrar exatamente quais contrastes têm equivalência física, relativa ou apenas pareamento numérico. Não agrupar paredes e cantos em um único resultado principal antes de verificar diferenças por condição.

Pré-registrar famílias de testes, método dos intervalos e tratamento de episódios sem primeira comida. Intervalos bootstrap pareados são condicionais aos checkpoints e cenários; corrigir multiplicidade sobre a família completa declarada. Não converter um intervalo descritivo em p-valor nem escolher contrastes somente após observar os resultados. Uma alegação de não inferioridade exige margem prática e análise de poder definidas antes da campanha.

## Recompensa, desconto e truncamento

O código histórico usa +1 por alimento, −0,01 por passo sem alimento, −1 na colisão e `gamma=0.98`. O retorno acumulado sem desconto de 200 passos sem alimento é −2. A sequência isolada descontada vale:

```text
G_200 = -0.01 * (1 - 0.98**200) / (1 - 0.98) ≈ -0.4912
```

Uma colisão imediata vale −1. Logo, comparar apenas −2 com −1 não demonstra que a DQN prefira colidir. O alvo do treinador ainda faz bootstrap quando o episódio é truncado; esses cálculos ilustrativos não equivalem ao alvo completo aprendido.

A distinção entre término e truncamento deve ser mantida. O treinador histórico passa `terminated` ao replay, e o alvo multiplica o próximo valor por `1 - terminated`. A orientação oficial do [Gymnasium sobre limites de tempo](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) recomenda bootstrap após truncamento externo e sua interrupção em término real. Se o limite for parte intrínseca de uma tarefa de horizonte finito, a definição da tarefa e a observação de tempo restante precisam ser revistas em um experimento separado.

Hipóteses futuras, sem modificar os modelos atuais:

1. **Sensibilidade à penalização por passo:** comparar uma escala pré-definida com novas seeds de treinamento, mesma arquitetura/orçamento e score held-out como endpoint primário. Não usar somente a recompensa cujo valor foi alterado para declarar melhoria.
2. **Sensibilidade a gamma e horizonte:** fixar famílias de valores e distinguir efeitos do desconto, truncamento e exploração; conservar registros completos de seeds, replay e RNG em novos checkpoints se retomada exata for um requisito.
3. **Estagnação:** testar se diagnósticos de ciclos e tempo sem comida explicam os limites de passos observados. Não aplicar penalização de ciclo aos modelos históricos nem chamar o diagnóstico de exploração de recompensa comprovada.
4. **Representação parcial:** investigar memória temporal ou observação adicional em uma linhagem nova, pois o vetor relativo 11D não descreve toda a geometria do corpo. Controlar mudanças de representação, arquitetura e treino separadamente.

Qualquer alteração de recompensa ou treinamento precisa de protocolo, orçamento e aprovação próprios. A Fase 5 congelada pode diagnosticar limites, mas não resolver por retreinamento sem uma decisão posterior.

## Mais inicializações e outros jogos

Após estabelecer os limites dos três checkpoints, planejar mais seeds de treinamento para estimar variabilidade do método, com separação entre seeds usadas para desenvolver a configuração e avaliação final held-out. Um baseline heurístico pode ser útil como referência explícita em protocolo novo, sem ser apresentado como descoberta da rede. Testes em mapas/regras inéditos e controle visual devem entrar em linhas independentes.

Para Pong, começar por simulador local Gymnasium com contrato e testes de física, uma política aleatória, um baseline simples documentado e uma rede treinada do zero. Reutilizar armazenamento, executor, limites e relatórios. Medir aprendizagem em Pong antes de comparar transferência. Uma futura comparação entre inicialização nova e compartilhada deve equalizar orçamento e evitar vazamento de dados entre treino e teste.

Um modelo de mundo é uma etapa posterior: prever observação/estado latente, recompensa e continuação, medir erro em trajetórias held-out e só então testar controle baseado no modelo contra referências de mesmo orçamento. O artigo primário [DreamerV3 — Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2) demonstra aprendizagem com modelos de mundo em mais de 150 tarefas e configuração comum; isso fundamenta uma direção de pesquisa, sem demonstrar que os pesos Snake existentes resolvam jogos desconhecidos. A arquitetura, o custo e a escala experimental do artigo exigem avaliação própria antes de adoção.

Pesquisa bibliográfica e recuperação de documentos podem ajudar a formular hipóteses e localizar resultados. Elas não constituem atualização dos pesos nem evidência de habilidade nova. O critério de progresso continuará sendo desempenho medido em experimentos autorizados, reproduzíveis e preservados.
