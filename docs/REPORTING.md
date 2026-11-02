# Relatórios descritivos reutilizáveis

`arena.evaluation.reporting` transforma `episodes.csv` existente em resumo JSON,
tabela Markdown e, opcionalmente, gráficos PNG/SVG. Não importa PyTorch, não
desserializa modelos, não joga partidas e não faz testes inferenciais pós-hoc.

Exemplo, usando a cópia preservada da Fase 4 e um destino novo:

```powershell
python -m arena.evaluation.reporting --source experiments/archive/verified/13110740130 --out experiments/reports/phase4-descriptive-v1 --expected-csv-sha256 89bac44718b347be85862cd66f162f99b2c184e3c01eb9f49a4e1ae0c4f79f64
```

O módulo funciona somente com a biblioteca padrão de Python. Para gráficos:

```powershell
python -m pip install -c requirements-reporting.txt -e '.[reporting]'
python -m arena.evaluation.reporting --source experiments/archive/verified/13110740130 --out experiments/reports/phase4-plots-v1 --plots --expected-csv-sha256 89bac44718b347be85862cd66f162f99b2c184e3c01eb9f49a4e1ae0c4f79f64
```

A API é `create_report(source, output, plots=False, expected_csv_sha256=None)`.
A origem deve conter `SHA256SUMS`, que será verificado integralmente. O CSV é
lido de bytes limitados e verificados; a opção de SHA independente permite
ancorá-lo em um catálogo confiável. Um manifesto distribuído junto com os dados,
sozinho, comprova integridade interna, não autenticidade independente.

São aceitos os esquemas de avaliações das Fases 2–4 e do microteste sintético da
Fase 5. A chave cenário/política/seed deve ser única; campos numéricos precisam
ser finitos, score não negativo, passos positivos e primeiro alimento/desfechos
consistentes. Cabeçalhos duplicados e campos geométricos incompletos são
recusados. Dados de treino da Fase 1 têm outro esquema e não são aceitos.

Os grupos preservam cenário e política, com média, mediana e desvio padrão
amostral de score, recompensa, passos e alimento/100 passos; incluem taxa de
obtenção de alimento, censura do primeiro alimento e repetição de estados quando
essas instrumentações existem. Ausência aparece como `null`/`n/a`, nunca zero.
Eficiência média é a média das eficiências por episódio. O tempo até alimento
médio entre sucessos é sujeito a seleção e deve ser lido junto à censura.

O gráfico agrega score por política sobre **as condições e episódios observados**,
com essa agregação explicitada e sem barras de erro inferenciais. Condições e
pesos fixos podem ser correlacionados; o gráfico não prova equivalência,
transferência ou independência de treinamentos. O detalhamento por cenário está
no JSON/Markdown. Repetição de estado não prova ciclo determinístico.

O destino deve ser novo e ficar fora da origem. São produzidos `summary.json`,
`report.md`, `provenance.json` e `SHA256SUMS`, além de `score-means.png/.svg` com
`--plots`. Proveniência registra hashes dos membros originais, código do
relatório/verificador e digest independente fornecido. A fonte é somente lida;
reexecutar no mesmo destino falha sem substituir saídas anteriores.
