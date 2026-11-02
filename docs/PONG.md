# Pong local — segundo ambiente da NOVA Neural Arena

**Status:** implementação técnica em PR separada, **sem treinar nenhuma rede**.
A Fase 5 de Snake permanece governada por protocolo/execução próprios.

## Contrato do ambiente

- Módulo: \`arena.envs.pong.PongEnv\`, implementa \`gymnasium.Env\`.
- Agente controla a **raquete esquerda**; raquete direita é um adversário
  determinístico incorporado ao ambiente (não é um modelo treinado).
- Tabuleiro padrão: 20×12, máximo 500 passos, alvo de 3 pontos.
- \`action_space = Discrete(3)\`: **0** parado, **1** subir, **2** descer.
- \`observation_space = Box(float32, shape=(8,))\`, valores normalizados:
  posição x/y da bola, velocidade x/y, posição y das duas raquetes e placar
  normalizado dos dois lados.
- **Recompensa \`pong-points-v1\`:** +1 por ponto da esquerda, −1 por ponto da
  direita, 0 em qualquer outro passo. A raquete automática não gera recompensa.
- **\`terminated=True\`** apenas quando um dos lados alcança o placar-alvo.
  **\`truncated=True\`** para o limite de passos sem vencedor. O truncamento
  não é vitória; não se deve usar \`winner\` se não houver término por pontos.
- \`reset(seed=N)\` semeia exclusivamente o RNG do ambiente. Oponente e colisões
  não consomem RNG. Os saques usam o RNG semeado, permitindo repetir partidas.
- Modo de renderização demonstrativo: \`ansi\`; não existe controle de conta
  externa, browser ou servidor de jogo.

## Uso sem treinamento

Após instalar as dependências Python 3.11 do laboratório:

\`\`\`bash
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-ci.txt -e '.[dev]'
python -m pytest -q tests/test_pong.py tests/test_multigame.py
python -m arena.shared.multigame --output experiments/demos/pong-local-001 --seed 123 --max-steps 100
\`\`\`

Use **pasta de saída nova**. Se já existir, o comando recusa sobrescrita.
A demonstração usa ações **ALEATÓRIAS**, não DQN, e não equivale a aprendizagem.

Saídas: \`trajectory.jsonl\` (histórico das observações e ações),
\`summary.csv\`, \`report.json\` e \`SHA256SUMS\`. \`verify_demo(path)\` confere
a integridade dos três arquivos. Os hashes são internos; para autenticar sua
origem é necessário guardar o SHA-256 do pacote completo em registro confiável.

## Componentes comuns

\`arena.shared.multigame\` fornece:

- \`EnvironmentSpec\`: identidade de ambiente, formato da observação,
  quantidade de ações, versão de recompensa e limite;
- \`describe_environment(env)\`: identifica o contrato de Snake e Pong;
- \`check_checkpoint_contract(metadata, spec)\`: recusa checkpoint de jogo,
  observação, ação ou recompensa incompatível **antes** do carregamento;
- \`write_demo\` / \`verify_demo\`: gravação exclusiva, histórico de passos,
  metadados verificáveis e checagem de hashes.

Esses componentes não alteram checkpoints congelados ou avaliadores
históricos. O registro de trajetória em disco é **memória de dados**,
não memória neural nem transferência aprendida entre jogos.

## Próximo experimento independente (fora do fechamento atual)

Um treinador de Pong poderá ser desenvolvido futuramente, do zero, com
orçamento e protocolo separados; o sistema atual **não faz treinamento**.
O checkpoint futuro deverá incluir \`EnvironmentSpec\` e a versão da rede para
evitar que pesos Snake (observação 11D) sejam carregados indevidamente em Pong
(observação 8D). Não se presume transferência de habilidades entre jogos.
