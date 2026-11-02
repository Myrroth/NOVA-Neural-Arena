# Pong local — segundo ambiente da NOVA Neural Arena

**Status:** implementação técnica, **sem treinar nenhuma rede**. A árvore atual
usa `pong-local-v2`; a versão `pong-local-v1` da PR #7 permanece recuperável no
commit `c41f96fe3a6272e4881339afa30cbb4f28d19896` e no histórico preservado.
A Fase 5 de Snake permanece governada por protocolo/execução próprios.

## Contrato do ambiente

- Módulo: `arena.envs.pong.PongEnv`, implementa `gymnasium.Env`.
- Agente controla a **raquete esquerda**; raquete direita é um adversário
  determinístico incorporado ao ambiente (não é um modelo treinado). Na v2,
  reage nos passos 1, 3, 5, ... e espera nos passos pares; quando reage, move
  no máximo uma célula em direção à bola ou ao centro. Essa cadência é fixa,
  reiniciada pelo reset, e não consome números aleatórios.
- Tabuleiro padrão: 20×12, máximo 500 passos, alvo de 3 pontos.
- `action_space = Discrete(3)`: **0** parado, **1** subir, **2** descer.
- `observation_space = Box(float32, shape=(8,))`, valores normalizados:
  posição x/y da bola, velocidade x/y, posição y das duas raquetes e placar
  normalizado dos dois lados.
- **Recompensa `pong-points-v1`:** +1 por ponto da esquerda, −1 por ponto da
  direita, 0 em qualquer outro passo. A raquete automática não gera recompensa.
  A identidade do ambiente passou a `pong-local-v2` porque a dinâmica mudou;
  o nome da recompensa continua igual porque seus valores e significado foram
  preservados. O contrato recusa checkpoints identificados como `pong-local-v1`.
- **`terminated=True`** apenas quando um dos lados alcança o placar-alvo.
  **`truncated=True`** para o limite de passos sem vencedor. O truncamento
  não é vitória; não se deve usar `winner` se não houver término por pontos.
- `reset(seed=N)` semeia exclusivamente o RNG do ambiente. Oponente e colisões
  não consomem RNG. Os saques usam o RNG semeado, permitindo repetir partidas.
- Modo de renderização demonstrativo: `ansi`; não existe controle de conta
  externa, browser ou servidor de jogo.

## Uso sem treinamento

Após instalar as dependências Python 3.11 do laboratório:

```bash
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-ci.txt -e '.[dev]'
python -m pytest -q tests/test_pong.py tests/test_multigame.py
python -m arena.shared.multigame --output experiments/demos/pong-local-001 --seed 123 --max-steps 100
```

Use **pasta de saída nova**. Se já existir, o comando recusa sobrescrita.
A demonstração usa ações **ALEATÓRIAS**, não DQN, e não equivale a aprendizagem.

Saídas: `trajectory.jsonl` (histórico das observações e ações),
`summary.csv`, `report.json` e `SHA256SUMS`. `verify_demo(path)` confere
a integridade dos três arquivos. Os hashes são internos; para autenticar sua
origem é necessário guardar o SHA-256 do pacote completo em registro confiável.

## Componentes comuns

`arena.shared.multigame` fornece:

- `EnvironmentSpec`: identidade de ambiente, formato da observação,
  quantidade de ações, versão de recompensa e limite;
- `describe_environment(env)`: identifica o contrato de Snake e Pong;
- `check_checkpoint_contract(metadata, spec)`: recusa checkpoint de jogo,
  observação, ação ou recompensa incompatível **antes** do carregamento;
- `write_demo` / `verify_demo`: gravação exclusiva, histórico de passos,
  metadados verificáveis e checagem de hashes.

Esses componentes não alteram checkpoints congelados ou avaliadores
históricos. O registro de trajetória em disco é **memória de dados**,
não memória neural nem transferência aprendida entre jogos.

## Próximo experimento independente (fora do fechamento atual)

Um treinador de Pong poderá ser desenvolvido futuramente, do zero, com
orçamento e protocolo separados; o sistema atual **não faz treinamento**.
O checkpoint futuro deverá incluir `EnvironmentSpec` e a versão da rede para
evitar que pesos Snake (observação 11D) sejam carregados indevidamente em Pong
(observação 8D). Não se presume transferência de habilidades entre jogos.

## Correção de alcançabilidade da v2

Na v1 padrão 20×12, a raquete direita se movia a cada passo, com a mesma
velocidade vertical da bola e alcance de três células. A auditoria de transições
encontrou ciclos de 352 passos para as quatro velocidades possíveis de saque:
mesmo uma esquerda que rebate perfeitamente nunca marcava um ponto. Os testes
que forçavam uma bola próxima da direita verificavam a regra de pontuação, mas
não sua alcançabilidade por ações legais a partir do reset.

A v2 reduz somente a frequência de reação do adversário para um passo a cada
dois. O novo teste começa em `reset(seed)`, lê a observação pública e usa uma
fixture explícita que acompanha a altura da bola com ações legais. Ele não
altera bola, raquetes ou RNG durante a trajetória e alcança recompensa +1.
Essa fixture serve exclusivamente para provar que o objetivo é alcançável;
não é uma política aprendida nem evidência de eficácia de treinamento.

O teste da cadência também verifica ausência de sorteios de RNG e os testes
existentes continuam cobrindo determinismo, colisões, pontuação e Gymnasium.
Snake, seus checkpoints e o executor congelado da Fase 5 não são alterados.
