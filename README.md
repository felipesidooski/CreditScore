# Credit Score - PUCPR

Projeto de Modelagem Avancada de Dados para predicao de inadimplencia em uma
base de credit scoring.

O objetivo e estimar se um cliente quitara sua divida integralmente
(`TARGET=0`) ou se entrara em inadimplencia (`TARGET=1`). A entrega compara os
modelos obrigatorios da rubrica, KNN e Arvore de Decisao, e tambem apresenta
experimentos avancados com XGBoost, LightGBM, stacking, selecao de atributos,
Optuna, calibracao e reamostragem.

*Nota/Observação*: Todas as classes e métodos foram escritas usando docstring baseado no
modelo google. A revisão, melhoria e padronização das mesmas foi realizada utilizando
auxilios de IA (claude).

## Como usar no Colab

No Google Colab, a execucao recomendada e clonar o repositorio, instalar as
dependencias e carregar este README como Markdown para servir como roteiro da
apresentacao.

```bash
git clone https://github.com/felipesidooski/CreditScore.git
cd CreditScore
pip install -r requirements.txt
```

Para renderizar este README dentro do proprio notebook Colab:

```python
from IPython.display import Markdown, display

display(Markdown(open("README.md", encoding="utf-8").read()))
```

Para carregar os modelos prontos e gerar predicoes:

```bash
python scripts/predict_with_artifacts.py \
  --artifact models/importancexgboost_top50.joblib \
  --input data/train.csv \
  --output reports/prediction_examples_importance_top50.csv \
  --n-rows 20
```

Para treinar novamente apenas os dois modelos campeoes, sem repetir toda a
busca experimental:

```bash
python scripts/train_champion_models.py \
  --train-path data/train.csv \
  --output-dir models
```

Em ambiente local, execute os comandos com a venv ativa. Caso a venv esteja na
pasta `CREDITSCORE`, acima do repositorio `GitCreditScore`, tambem e possivel
executar pela raiz do repositorio com:

```bash
../bin/python scripts/train_champion_models.py
../bin/python scripts/predict_with_artifacts.py --n-rows 20
```

## Estrutura do projeto

```text
.
├── data/
│   └── train.csv
├── legacy/
│   ├── archive/
│   │   └── credit_score_experiments_full.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── original_models.py
│   ├── enhanced_models.py
│   ├── advanced_models.py
│   ├── ensemble_benchmarks.py
│   ├── resampling.py
│   ├── optuna_xgboost.py
│   ├── feature_selection.py
│   ├── stacking.py
│   └── target_encoding.py
├── models/
│   ├── importancexgboost_top50.joblib
│   └── optunaxgboost.joblib
├── reports/
│   ├── prediction_examples_importance_top50.csv
│   ├── prediction_examples_optuna_xgboost.csv
│   └── tables/
├── scripts/
│   ├── predict_with_artifacts.py
│   └── train_champion_models.py
└── src/
    └── creditscore/
        ├── config.py
        ├── inference.py
        ├── preprocessing.py
        └── training.py
```

## Dataset

Arquivo principal:

```text
data/train.csv
```

Resumo da base:

| Item | Valor |
|---|---:|
| Instancias | 92.106 |
| Atributos totais | 70 |
| TARGET=0 | 83.298 (90,44%) |
| TARGET=1 | 8.808 (9,56%) |
| NaN explicito | 0% |
| Codigos sentinela | -99, -9998, -9999 |

O desbalanceamento e forte: apenas `9,56%` da base pertence a classe
inadimplente. Por isso, acuracia nao foi usada como criterio principal. Um
modelo trivial que sempre prediz `TARGET=0` ja atinge `90,44%` de acuracia, mas
tem `recall=0` para inadimplentes.

Tabela de missing/sentinelas:

```text
reports/tables/missing_sentinel_report.csv
```

Principais colunas com sentinelas:

| Coluna | % sentinelas | Decisao |
|---|---:|---|
| ANOSULTIMADECLARACAOPAGAR | 90,66% | Removida |
| ANOSULTIMARESTITUICAO | 83,52% | Removida |
| ANOSULTIMADECLARACAO | 65,30% | Mantida com flag |
| Bloco CASA | ~62,08% | Mantido com features de ausencia |

## Pre-processamento

O pre-processamento final foi separado em duas classes principais.

### `CreditScorePreprocessor`

Arquivo:

```text
src/creditscore/preprocessing.py
```

Responsabilidades:

- substituir sentinelas `-99`, `-9998`, `-9999` por `NaN`;
- criar `GRUPO_CASA_AUSENTE`;
- criar `DECLARACAO_AUSENTE`;
- remover `HS_CPF`, pois e identificador;
- remover `ORIENTACAO_SEXUAL` e `RELIGIAO`, pois sao variaveis sensiveis;
- remover `ANOSULTIMARESTITUICAO` e `ANOSULTIMADECLARACAOPAGAR`, pois possuem
  ausencia extrema.

Campos de entrada esperados:

- dataframe no mesmo formato do `train.csv`;
- a coluna `TARGET` pode existir durante treino ou estar ausente durante
  inferencia.

Saida esperada:

- dataframe limpo, com sentinelas convertidos para `NaN`, colunas proibidas
  removidas e flags deterministicas adicionadas.

### `CreditScoreFeatureEngineer`

Responsabilidades:

- criar contagem e percentual de campos ausentes;
- criar agregacoes de ausencia para CASA, CEP e MUNICIPIO;
- criar transformacoes logaritmicas;
- criar flags de zero para variaveis de contagem;
- criar razoes entre renda, CEP, CASA, IDH, PIB e veiculos.

Exemplos de features criadas:

```text
QTD_CAMPOS_AUSENTES
PCT_CAMPOS_AUSENTES
CASA_QTD_AUSENTES
LOG_ESTIMATIVARENDA
RAZAO_RENDA_ESTIMADA_CEP
RAZAO_RENDA_ESTIMADA_IDH
TOTAL_VEICULOS_MUNICIPIO
RAZAO_RENDA_ESTIMADA_VEICULOS
```

### `CreditScoreDataPipeline`

Executa a sequencia:

```text
train.csv bruto
  -> CreditScorePreprocessor
  -> CreditScoreFeatureEngineer
  -> X/y para treino ou X para predicao
```

Essa etapa e deterministica e nao aprende estatisticas globais. Isso reduz o
risco de data leakage e permite usar exatamente o mesmo preparo em treino e
inferencia.

## Protocolo experimental

O protocolo foi definido antes do treinamento para evitar comparacoes
inconsistentes entre modelos.

| Item | Definicao utilizada |
|---|---|
| Seed global | `42` |
| Split holdout | 80% treino e 20% teste |
| Estratificacao | `stratify=TARGET`, mantendo a proporcao de inadimplentes em treino e teste |
| Validacao cruzada | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Baseline | `DummyClassifier`, para medir o ganho real sobre um modelo trivial |
| Metrica principal | `average_precision`, por ser mais adequada para classe positiva rara |
| Metricas reportadas | `accuracy`, `precision_1`, `recall_1`, `f1_1`, `roc_auc`, `average_precision`, matriz de confusao |
| Modelos obrigatorios | KNN e Arvore de Decisao com `GridSearchCV` |
| Modelos adicionais | XGBoost, LightGBM, RandomForest, ExtraTrees, HistGradientBoosting, Stacking, SMOTE, NearMiss, Optuna e selecao de atributos |

O criterio principal de selecao foi o `average_precision`, pois o objetivo de
negocio e ranquear e identificar clientes com maior probabilidade de
inadimplencia. O `ROC-AUC` foi mantido como metrica complementar para avaliar a
separacao geral entre adimplentes e inadimplentes.

## Modelos implementados

Os experimentos historicos foram separados por familia de metodo dentro da
pasta `legacy/`. Cada arquivo concentra uma etapa ou uma familia de modelos
testados durante a evolucao do projeto.

| Etapa | Arquivo principal | O que foi processado | Saida/resultados |
|---|---|---|---|
| AED e estatisticas iniciais | `legacy/eda.py` | Analises exploratorias, distribuicoes, missing/sentinelas e tabelas de apoio | Relatorios e graficos da AED |
| Limpeza deterministica | `legacy/preprocessing.py` e `src/creditscore/preprocessing.py` | Substituicao de sentinelas, remocao de colunas proibidas e criacao de flags | Base preparada para treino e predicao |
| Pipelines sklearn | `legacy/pipelines.py` | `ColumnTransformer`, imputacao, normalizacao e pipelines com classificadores | Pipelines usados nos modelos KNN e Arvore |
| Validacao de decisoes tecnicas | `legacy/decision_validation.py` | Testes de scaler, manutencao/remocao de atributos e tratamento de ausencia | Comparacao empirica das escolhas de pre-processamento |
| Modelos obrigatorios | `legacy/original_models.py` | KNN e Arvore de Decisao com `GridSearchCV` | `reports/tables/final_holdout_metrics.csv` |
| Modelos com engenharia de atributos | `legacy/enhanced_models.py` | KNN e Arvore com features logaritmicas, razoes e agregacoes | `reports/tables/enhanced_final_holdout_metrics.csv` |
| Modelos advanced | `legacy/advanced_models.py` | KNN, Arvore, calibracao e thresholds alternativos | `reports/tables/advanced_final_holdout_metrics.csv` |
| Reamostragem | `legacy/resampling.py` | SMOTE e NearMiss com KNN/Arvore | `reports/tables/resampling_holdout_metrics.csv` |
| Ensembles e boosting | `legacy/ensemble_benchmarks.py` | XGBoost, LightGBM, HistGradientBoosting, RandomForest e ExtraTrees | `reports/tables/benchmark_ensemble_holdout_metrics.csv` |
| Boosting nativo | `legacy/ensemble_benchmarks.py` | XGBoost e LightGBM com bibliotecas nativas quando disponiveis | `reports/tables/native_boosting_holdout_metrics.csv` |
| Selecao de atributos | `legacy/feature_selection.py` | XGBoost com Top 30, Top 50, Top 70 e todas as features | `reports/tables/importance_selection_holdout_metrics.csv` |
| Otimizacao bayesiana | `legacy/optuna_xgboost.py` | Busca de hiperparametros com Optuna para XGBoost | `reports/tables/optuna_xgboost_holdout_metrics.csv` |
| Stacking | `legacy/stacking.py` | Combinacao de modelos com meta-modelo | `reports/tables/stacking_holdout_metrics.csv` |
| Treino final dos campeoes | `scripts/train_champion_models.py` e `src/creditscore/training.py` | Treino dos dois modelos escolhidos para entrega | `models/*.joblib` |
| Predicao com modelos salvos | `scripts/predict_with_artifacts.py` e `src/creditscore/inference.py` | Carregamento dos artefatos e geracao de predicoes | `reports/prediction_examples_*.csv` |

O script completo original foi preservado apenas como referencia historica em:

```text
legacy/archive/credit_score_experiments_full.py
```

As classes finais utilizadas para treinar e consumir os modelos salvos estao
nos modulos:

```text
src/creditscore/training.py
src/creditscore/inference.py
```

### Como utilizar cada arquivo principal

Os arquivos em `src/` e `scripts/` representam a versao final e limpa do
projeto. Os arquivos em `legacy/` preservam os experimentos completos que foram
executados durante a busca por melhoria de desempenho.

| Objetivo | Comando no Colab ou terminal | Resultado esperado |
|---|---|---|
| Exibir ajuda do script legado completo | `python legacy/archive/credit_score_experiments_full.py --help` | Lista de flags disponiveis para reproduzir os experimentos |
| Executar apresentacao da base e AED | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --eda` | Tabelas e graficos de AED em `outputs/` |
| Validar decisoes de pre-processamento | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --validate-decisions` | Comparacoes de scaler, flags e atributos mantidos/removidos |
| Rodar KNN e Arvore obrigatorios | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --model` | Resultados de `GridSearchCV` e holdout para KNN/Arvore |
| Rodar modelos enhanced | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --model --enhanced-model` | Resultados com engenharia de atributos adicional |
| Rodar modelos advanced | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model` | Resultados com features advanced, thresholds e calibracao |
| Rodar ensembles e boosting | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --benchmark-ensembles` | Comparacao com RandomForest, ExtraTrees, HistGradientBoosting, XGBoost e LightGBM |
| Rodar reamostragem | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --resampling-models` | Comparacao de SMOTE e NearMiss |
| Rodar XGBoost/LightGBM nativos | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --native-boosting` | Boosting preservando tratamento nativo de `NaN` |
| Rodar Optuna XGBoost | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --optuna-xgboost` | Busca bayesiana e melhores parametros do XGBoost |
| Rodar selecao por importancia | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --importance-selection` | Ranking de features e testes Top 30/50/70/all |
| Rodar stacking | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --stacking-models` | Resultados do `StackingClassifier` |
| Rodar todos os extras avancados | `python legacy/archive/credit_score_experiments_full.py --data-path data/train.csv --output-dir outputs --advanced-model --advanced-extras` | Native boosting, Optuna, importance selection, stacking e target encoding |
| Retreinar somente os campeoes | `python scripts/train_champion_models.py --train-path data/train.csv --output-dir models` | Gera `models/importancexgboost_top50.joblib` e `models/optunaxgboost.joblib` |
| Usar modelo pronto para predicao | `python scripts/predict_with_artifacts.py --artifact models/importancexgboost_top50.joblib --input data/train.csv --output reports/prediction_examples_importance_top50.csv --n-rows 20` | CSV com probabilidades e classes previstas |

As tabelas versionadas em `reports/tables/` foram geradas a partir dessas
execucoes historicas. Por isso, o Colab nao precisa retreinar todos os modelos
para apresentar os resultados; basta carregar o README e, no final, executar os
modelos pre-treinados para demonstrar predicoes.

## Requisitos de projeto

Este projeto exige a execução com apresentado da base de dados, AED, pre-processamento,
treinamento e protocolo experimental. Estrutura da entrega:

| Requisito do projeto | Onde aparece no projeto | Observacao |
|---|---|---|
| Contexto do problema | `README.md` e `notebooks/credit_score.ipynb` | Problema de credit scoring com `TARGET=0/1` |
| Quantidade de instancias e atributos | `README.md`, notebook e `reports/tables/missing_sentinel_report.csv` | 92.106 instancias e 70 colunas no bruto |
| Percentual de faltantes | `reports/tables/missing_sentinel_report.csv` | Considera `NaN` explicito e sentinelas |
| Descricao de 10+ atributos | `notebooks/credit_score.ipynb` | Descricao textual de atributos centrais |
| 15 analises univariadas | `notebooks/credit_score.ipynb` e `legacy/eda.py` | Estrutura questao/hipotese -> analise/discussao |
| 5 analises multivariadas | `notebooks/credit_score.ipynb` e `legacy/eda.py` | Estrutura questao/hipotese -> analise/discussao |
| Pre-processamento com `ColumnTransformer` | `legacy/pipelines.py` | Imputacao, normalizacao e OHE quando aplicavel |
| Pipeline sklearn | `legacy/original_models.py` e `legacy/pipelines.py` | `Pipeline([("preprocess", ...), ("clf", ...)])` |
| Comparacao KNN vs Arvore | `legacy/original_models.py` | Modelos obrigatorios com `GridSearchCV` |
| Protocolo experimental | Notebook e README | Seed, split, CV, metricas e criterio de selecao |
| Uso de modelos prontos | `models/*.joblib` e `scripts/predict_with_artifacts.py` | Permite predicao sem retreinar |

### `ChampionModelTrainer`

Arquivo:

```text
src/creditscore/training.py
```

Responsabilidades:

- carregar `train.csv`;
- aplicar o pipeline deterministico;
- treinar `ImportanceXGBoost_top50`;
- treinar `OptunaXGBoost`;
- salvar os artefatos `.joblib` em `models/`.

Metodos principais:

| Metodo | O que faz | Resultado esperado |
|---|---|---|
| `prepare_training_data()` | Carrega e prepara o CSV bruto | `X`, `y` prontos para treino |
| `load_top_features()` | Carrega ranking SHAP/importancia | lista com top-N features |
| `load_optuna_params()` | Carrega melhores parametros do Optuna | dicionario de hiperparametros |
| `train_importance_xgboost_top50()` | Treina o melhor modelo por AP | artefato treinado |
| `train_optuna_xgboost()` | Treina o melhor modelo por ROC-AUC | artefato treinado |
| `train_and_save_all()` | Treina e salva os dois campeoes | arquivos `.joblib` |

### `CreditScorePredictor`

Responsabilidades:

- carregar um artefato `.joblib`;
- preparar dados brutos com o mesmo pipeline do treino;
- alinhar colunas esperadas pelo modelo;
- calcular probabilidade de inadimplencia;
- aplicar threshold operacional;
- retornar predicao final.

Metodos principais:

| Metodo | O que faz | Resultado esperado |
|---|---|---|
| `prepare()` | Prepara e alinha colunas | dataframe pronto para o modelo |
| `predict_proba()` | Calcula risco de `TARGET=1` | serie com probabilidades |
| `predict()` | Calcula probabilidade e classe final | dataframe com modelo, probabilidade, threshold e predicao |

## Resultados gerais

### Como interpretar as metricas

As metricas abaixo aparecem nas tabelas de resultados. Como a base e
desbalanceada, nenhuma metrica deve ser lida isoladamente.

### Matriz de confusao

A matriz de confusao compara a classe real com a classe prevista pelo modelo.
Neste projeto, a classe positiva e `TARGET=1`, ou seja, inadimplente.

| Termo | Nome | Significado no projeto |
|---|---|---|
| `TN` | True Negative | Cliente era adimplente (`TARGET=0`) e o modelo previu adimplente |
| `FP` | False Positive | Cliente era adimplente, mas o modelo sinalizou como inadimplente |
| `FN` | False Negative | Cliente era inadimplente, mas o modelo previu como adimplente |
| `TP` | True Positive | Cliente era inadimplente e o modelo sinalizou como inadimplente |

Em credito, `FN` costuma ser o erro mais perigoso, pois representa conceder ou
aprovar credito para um cliente que de fato era inadimplente. `FP` representa
um falso alerta: o modelo sinaliza risco em um cliente que era adimplente, o que
pode gerar analise manual desnecessaria ou perda de oportunidade.

As principais metricas podem ser lidas pelas formulas:

```text
accuracy    = (TP + TN) / (TP + TN + FP + FN)
precision_1 = TP / (TP + FP)
recall_1    = TP / (TP + FN)
f1_1        = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
```

| Metrica | O que representa | Como interpretar neste projeto |
|---|---|---|
| `threshold` | Ponto de corte usado para transformar probabilidade em classe final | Se `probabilidade_inadimplencia >= threshold`, o cliente e classificado como risco (`TARGET=1`) |
| `accuracy` | Percentual total de acertos | Deve ser lida com cuidado, pois o baseline ja atinge 90,44% prevendo todos como adimplentes |
| `precision_1` | Entre os clientes marcados como inadimplentes, quantos realmente eram inadimplentes | Mede a qualidade dos alertas de risco; baixa precision significa muitos falsos positivos |
| `recall_1` | Entre todos os inadimplentes reais, quantos foram encontrados | Mede a capacidade de detectar maus pagadores; e uma metrica critica para risco de credito |
| `f1_1` | Media harmonica entre `precision_1` e `recall_1` | Resume o equilibrio entre encontrar inadimplentes e nao gerar alertas demais |
| `roc_auc` | Capacidade geral de separar adimplentes e inadimplentes em diferentes thresholds | Valores acima de 0,50 indicam sinal preditivo real |
| `average_precision` ou `AP` | Area sob a curva Precision-Recall | Principal metrica de ranking para classe inadimplente em base desbalanceada |
| `experimento` | Familia de teste executada | Indica se o resultado veio do baseline, advanced, boosting, Optuna, stacking etc. |

O sufixo `_1` em `precision_1`, `recall_1` e `f1_1` indica que a metrica foi
calculada especificamente para a classe positiva `TARGET=1`. Essa escolha e
importante porque a classe 1 e a classe de interesse do problema: inadimplentes.

### Baseline

O `DummyClassifier` e o piso minimo do projeto. Ele sempre prediz a classe
majoritaria (`TARGET=0`). A acuracia fica alta, mas o modelo nao encontra
nenhum inadimplente.

| Modelo | Accuracy | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0,9044 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,0956 |

### Evolucao dos principais experimentos

| Modelo | Threshold | Accuracy | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN original | 0,50 | 0,9025 | 0,2167 | 0,0074 | 0,0143 | 0,6017 | 0,1393 |
| DecisionTree original | 0,50 | 0,5698 | 0,1381 | 0,6674 | 0,2288 | 0,6545 | 0,1546 |
| AdvancedDecisionTree calibrated | 0,10 | 0,6161 | 0,1478 | 0,6322 | 0,2396 | 0,6740 | 0,1775 |
| AdvancedXGBoost | 0,40 | 0,7988 | 0,2110 | 0,4030 | 0,2770 | 0,6970 | 0,1985 |
| OptunaXGBoost | 0,25 | 0,7947 | 0,2094 | 0,4132 | 0,2780 | 0,6990 | 0,2006 |
| ImportanceXGBoost_top50 | 0,35 | 0,7229 | 0,1800 | 0,5335 | 0,2692 | 0,6967 | 0,2020 |

### Ranking advanced consolidado no holdout

A tabela abaixo consolida os resultados avancados usados na comparacao final.
O arquivo completo com todos os resultados fica em
`reports/tables/comparison_all_experiments.csv`.

| Modelo | Threshold | Accuracy | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP | Experimento |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ImportanceXGBoost_top50@f1 | 0,3500 | 0,7229 | 0,1800 | 0,5335 | 0,2692 | 0,6967 | 0,2020 | importance_selection |
| ImportanceXGBoost_top50@cost | 0,3000 | 0,6415 | 0,1583 | 0,6368 | 0,2536 | 0,6967 | 0,2020 | importance_selection |
| ImportanceXGBoost_top50@f2 | 0,2000 | 0,4258 | 0,1259 | 0,8422 | 0,2191 | 0,6967 | 0,2020 | importance_selection |
| ImportanceXGBoost_top50@0.50 | 0,5000 | 0,8779 | 0,2739 | 0,1674 | 0,2078 | 0,6967 | 0,2020 | importance_selection |
| ImportanceXGBoost_topall@f1 | 0,4000 | 0,7993 | 0,2090 | 0,3944 | 0,2732 | 0,6973 | 0,2014 | importance_selection |
| ImportanceXGBoost_topall@cost | 0,2500 | 0,5421 | 0,1410 | 0,7440 | 0,2371 | 0,6973 | 0,2014 | importance_selection |
| ImportanceXGBoost_topall@f2 | 0,2000 | 0,4184 | 0,1257 | 0,8530 | 0,2191 | 0,6973 | 0,2014 | importance_selection |
| ImportanceXGBoost_topall@0.50 | 0,5000 | 0,8781 | 0,2693 | 0,1600 | 0,2008 | 0,6973 | 0,2014 | importance_selection |
| OptunaXGBoost@f1 | 0,2500 | 0,7947 | 0,2094 | 0,4132 | 0,2780 | 0,6990 | 0,2006 | optuna_xgboost |
| OptunaXGBoost@f2 | 0,1500 | 0,5337 | 0,1402 | 0,7548 | 0,2364 | 0,6990 | 0,2006 | optuna_xgboost |
| OptunaXGBoost@cost | 0,1500 | 0,5337 | 0,1402 | 0,7548 | 0,2364 | 0,6990 | 0,2006 | optuna_xgboost |
| OptunaXGBoost@0.50 | 0,5000 | 0,9043 | 0,4898 | 0,0136 | 0,0265 | 0,6990 | 0,2006 | optuna_xgboost |
| ImportanceXGBoost_top70@f1 | 0,4000 | 0,7994 | 0,2100 | 0,3973 | 0,2747 | 0,6951 | 0,1997 | importance_selection |
| ImportanceXGBoost_top70@cost | 0,2500 | 0,5411 | 0,1401 | 0,7389 | 0,2355 | 0,6951 | 0,1997 | importance_selection |
| ImportanceXGBoost_top70@f2 | 0,2000 | 0,4218 | 0,1255 | 0,8451 | 0,2185 | 0,6951 | 0,1997 | importance_selection |
| ImportanceXGBoost_top70@0.50 | 0,5000 | 0,8770 | 0,2652 | 0,1612 | 0,2005 | 0,6951 | 0,1997 | importance_selection |
| NativeXGBoost@f1 | 0,2500 | 0,7983 | 0,2099 | 0,4012 | 0,2756 | 0,6982 | 0,1984 | native_boosting |
| NativeXGBoost@f2 | 0,1500 | 0,5374 | 0,1404 | 0,7491 | 0,2365 | 0,6982 | 0,1984 | native_boosting |
| NativeXGBoost@cost | 0,1500 | 0,5374 | 0,1404 | 0,7491 | 0,2365 | 0,6982 | 0,1984 | native_boosting |
| NativeXGBoost@0.50 | 0,5000 | 0,9042 | 0,4722 | 0,0096 | 0,0189 | 0,6982 | 0,1984 | native_boosting |
| StackingBoosting@f1 | 0,6000 | 0,7752 | 0,2001 | 0,4506 | 0,2771 | 0,6982 | 0,1976 | stacking |
| StackingBoosting@0.50 | 0,5000 | 0,6545 | 0,1609 | 0,6198 | 0,2555 | 0,6982 | 0,1976 | stacking |
| StackingBoosting@cost | 0,5000 | 0,6545 | 0,1609 | 0,6198 | 0,2555 | 0,6982 | 0,1976 | stacking |
| StackingBoosting@f2 | 0,4500 | 0,5726 | 0,1471 | 0,7230 | 0,2445 | 0,6982 | 0,1976 | stacking |
| NativeLightGBM@f1 | 0,2500 | 0,8010 | 0,2138 | 0,4035 | 0,2795 | 0,6980 | 0,1973 | native_boosting |
| NativeLightGBM@f2 | 0,1500 | 0,5587 | 0,1431 | 0,7247 | 0,2390 | 0,6980 | 0,1973 | native_boosting |
| NativeLightGBM@cost | 0,1500 | 0,5587 | 0,1431 | 0,7247 | 0,2390 | 0,6980 | 0,1973 | native_boosting |
| NativeLightGBM@0.50 | 0,5000 | 0,9038 | 0,4074 | 0,0125 | 0,0242 | 0,6980 | 0,1973 | native_boosting |
| ImportanceXGBoost_top30@f1 | 0,4000 | 0,7954 | 0,2076 | 0,4047 | 0,2744 | 0,6943 | 0,1948 | importance_selection |
| ImportanceXGBoost_top30@cost | 0,3000 | 0,6397 | 0,1572 | 0,6345 | 0,2520 | 0,6943 | 0,1948 | importance_selection |
| ImportanceXGBoost_top30@f2 | 0,2500 | 0,5396 | 0,1395 | 0,7378 | 0,2346 | 0,6943 | 0,1948 | importance_selection |
| ImportanceXGBoost_top30@0.50 | 0,5000 | 0,8744 | 0,2600 | 0,1697 | 0,2054 | 0,6943 | 0,1948 | importance_selection |
| AdvancedDecisionTreeCalibrated_sigmoid@threshold_f1 | 0,1000 | 0,6161 | 0,1478 | 0,6322 | 0,2396 | 0,6740 | 0,1775 | advanced |
| AdvancedDecisionTreeCalibrated_sigmoid@threshold_f2 | 0,1000 | 0,6161 | 0,1478 | 0,6322 | 0,2396 | 0,6740 | 0,1775 | advanced |
| AdvancedDecisionTreeCalibrated_sigmoid@threshold_cost | 0,1000 | 0,6161 | 0,1478 | 0,6322 | 0,2396 | 0,6740 | 0,1775 | advanced |
| AdvancedDecisionTreeCalibrated_sigmoid@0.50 | 0,5000 | 0,9044 | 0,0000 | 0,0000 | 0,0000 | 0,6740 | 0,1775 | advanced |
| AdvancedDecisionTreeCalibrated_isotonic@threshold_f1 | 0,1000 | 0,6421 | 0,1517 | 0,5970 | 0,2419 | 0,6736 | 0,1760 | advanced |
| AdvancedDecisionTreeCalibrated_isotonic@threshold_f2 | 0,1000 | 0,6421 | 0,1517 | 0,5970 | 0,2419 | 0,6736 | 0,1760 | advanced |
| AdvancedDecisionTreeCalibrated_isotonic@threshold_cost | 0,1000 | 0,6421 | 0,1517 | 0,5970 | 0,2419 | 0,6736 | 0,1760 | advanced |
| AdvancedDecisionTreeCalibrated_isotonic@0.50 | 0,5000 | 0,9044 | 0,0000 | 0,0000 | 0,0000 | 0,6736 | 0,1760 | advanced |
| AdvancedDecisionTree@0.50 | 0,5000 | 0,8028 | 0,1994 | 0,3519 | 0,2545 | 0,6651 | 0,1669 | advanced |
| AdvancedDecisionTree@threshold_f1 | 0,4500 | 0,6700 | 0,1587 | 0,5698 | 0,2483 | 0,6651 | 0,1669 | advanced |
| AdvancedDecisionTree@threshold_cost | 0,4000 | 0,6445 | 0,1519 | 0,5925 | 0,2418 | 0,6651 | 0,1669 | advanced |
| AdvancedDecisionTree@threshold_f2 | 0,3500 | 0,4732 | 0,1262 | 0,7605 | 0,2164 | 0,6651 | 0,1669 | advanced |
| AdvancedKNN@threshold_f1 | 0,1500 | 0,7798 | 0,1741 | 0,3479 | 0,2321 | 0,6377 | 0,1578 | advanced |
| AdvancedKNN@threshold_f2 | 0,1000 | 0,6301 | 0,1418 | 0,5675 | 0,2269 | 0,6377 | 0,1578 | advanced |
| AdvancedKNN@threshold_cost | 0,1000 | 0,6301 | 0,1418 | 0,5675 | 0,2269 | 0,6377 | 0,1578 | advanced |
| AdvancedKNN@0.50 | 0,5000 | 0,9031 | 0,2143 | 0,0051 | 0,0100 | 0,6377 | 0,1578 | advanced |
| AdvancedDummyClassifier | 0,5000 | 0,9044 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,0956 | advanced |

### Arquivos com resultados completos

```text
reports/tables/comparison_all_experiments.csv
reports/tables/importance_selection_holdout_metrics.csv
reports/tables/optuna_xgboost_holdout_metrics.csv
reports/tables/benchmark_ensemble_holdout_metrics.csv
reports/tables/resampling_holdout_metrics.csv
reports/tables/native_boosting_holdout_metrics.csv
reports/tables/stacking_holdout_metrics.csv
```

## Analise dos melhores resultados

### Ranking por `Average Precision`

Considerando apenas a metrica principal de ranking da classe inadimplente, os
dois melhores resultados foram:

| Posicao | Modelo | Threshold | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ImportanceXGBoost_top50@f1 | 0,35 | 0,1800 | 0,5335 | 0,2692 | 0,6967 | 0,2020 |
| 2 | ImportanceXGBoost_topall@f1 | 0,40 | 0,2090 | 0,3944 | 0,2732 | 0,6973 | 0,2014 |

O `ImportanceXGBoost_top50` foi o melhor modelo por `Average Precision`. Isso
significa que ele foi o melhor para ordenar clientes por risco de inadimplencia,
que e o objetivo mais importante em um problema com classe positiva rara. Com
threshold `0,35`, ele encontrou `53,35%` dos inadimplentes reais
(`recall_1 = 0,5335`), com precisao de `18,00%` entre os clientes sinalizados
como risco.

O `ImportanceXGBoost_topall` teve desempenho muito proximo, com `AP = 0,2014`,
mas usa todas as features. Ele teve maior `precision_1` e maior `F1_1`, porem
menor `recall_1` que o Top 50. Como a diferenca de AP foi pequena e o Top 50
usa menos atributos, o modelo Top 50 foi escolhido como artefato principal para
ranking.

### Modelos campeoes escolhidos para entrega

Para a entrega com modelos pre-treinados, foram salvos dois artefatos:

| Modelo salvo | Arquivo | Motivo da escolha | Threshold | Accuracy | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ImportanceXGBoost_top50 | `models/importancexgboost_top50.joblib` | Melhor `Average Precision` | 0,35 | 0,7229 | 0,1800 | 0,5335 | 0,2692 | 0,6967 | 0,2020 |
| OptunaXGBoost | `models/optunaxgboost.joblib` | Melhor `ROC-AUC` e melhor F1 entre os modelos XGBoost otimizados | 0,25 | 0,7947 | 0,2094 | 0,4132 | 0,2780 | 0,6990 | 0,2006 |

O `ImportanceXGBoost_top50` e recomendado quando o objetivo principal e
identificar mais inadimplentes e produzir um ranking de risco mais sensivel. O
`OptunaXGBoost` e recomendado quando o objetivo e uma separacao global um pouco
melhor entre adimplentes e inadimplentes, com maior `F1_1` e maior `ROC-AUC`.

### Leitura dos dois campeoes

No `ImportanceXGBoost_top50`, a acuracia de `0,7229` e menor que a do baseline,
mas isso acontece porque o modelo deixa de prever tudo como adimplente e passa a
sinalizar risco. O ponto forte é o `recall_1 = 0,5335`: ele identifica mais da
metade dos inadimplentes reais. A precisao de `0,1800` significa que, entre os
clientes sinalizados como risco, 18% eram de fato inadimplentes. Esse valor pode
parecer baixo, mas deve ser comparado com a prevalencia original da classe
inadimplente, que e de apenas 9,56%, desta forma, estamos praticamente dobrando 
nossa precisão.

No `OptunaXGBoost`, a acuracia sobe para `0,7947` e o `F1_1` chega a `0,2780`,
o melhor equilibrio entre precisao e recall entre os modelos escolhidos. O
`ROC-AUC = 0,6990` indica a melhor capacidade geral de separacao entre as
classes. Em contrapartida, o `recall_1 = 0,4132` é menor que o do Top 50, ou
seja, ele detecta menos inadimplentes, mas gera alertas de risco mais
equilibrados.

## Modelos campeoes salvos

### 1. `ImportanceXGBoost_top50`

Arquivo:

```text
models/importancexgboost_top50.joblib
```

Justificativa:

- melhor `Average Precision` do projeto: `0,2020`;
- usa as 50 features mais importantes;
- melhor escolha quando o foco e ranking da classe inadimplente.

Metricas reportadas no holdout:

| Metrica | Valor |
|---|---:|
| Threshold | 0,35 |
| Accuracy | 0,7229 |
| Precision_1 | 0,1800 |
| Recall_1 | 0,5335 |
| F1_1 | 0,2692 |
| ROC-AUC | 0,6967 |
| Average Precision | 0,2020 |

### 2. `OptunaXGBoost`

Arquivo:

```text
models/optunaxgboost.joblib
```

Justificativa:

- melhor `ROC-AUC` entre os modelos XGBoost: `0,6990`;
- parametros escolhidos por busca bayesiana (https://pt.wikipedia.org/wiki/Infer%C3%AAncia_bayesiana) com Optuna;
- boa alternativa quando o foco é separacao global entre adimplentes e
  inadimplentes.

Metricas reportadas no holdout:

| Metrica | Valor |
|---|---:|
| Threshold | 0,25 |
| Accuracy | 0,7947 |
| Precision_1 | 0,2094 |
| Recall_1 | 0,4132 |
| F1_1 | 0,2780 |
| ROC-AUC | 0,6990 |
| Average Precision | 0,2006 |

## Exemplo pratico de predicao

Com o modelo `ImportanceXGBoost_top50`:

```bash
python scripts/predict_with_artifacts.py \
  --artifact models/importancexgboost_top50.joblib \
  --input data/train.csv \
  --output reports/prediction_examples_importance_top50.csv \
  --n-rows 20
```

Exemplo de saida:

| modelo | probabilidade_inadimplencia | threshold | predicao_target |
|---|---:|---:|---:|
| ImportanceXGBoost_top50 | 0,307660 | 0,35 | 0 |
| ImportanceXGBoost_top50 | 0,513340 | 0,35 | 1 |
| ImportanceXGBoost_top50 | 0,386877 | 0,35 | 1 |
| ImportanceXGBoost_top50 | 0,085400 | 0,35 | 0 |

Interpretacao:

- `probabilidade_inadimplencia`: risco estimado de `TARGET=1`;
- `threshold`: ponto de corte operacional do modelo;
- `predicao_target=1`: cliente sinalizado como maior risco;
- `predicao_target=0`: cliente nao sinalizado como risco pelo modelo.

Com o modelo `OptunaXGBoost`:

```bash
python scripts/predict_with_artifacts.py \
  --artifact models/optunaxgboost.joblib \
  --input data/train.csv \
  --output reports/prediction_examples_optuna_xgboost.csv \
  --n-rows 20
```

Exemplo de saida:

| modelo | probabilidade_inadimplencia | threshold | predicao_target |
|---|---:|---:|---:|
| OptunaXGBoost | 0,205738 | 0,25 | 0 |
| OptunaXGBoost | 0,326061 | 0,25 | 1 |
| OptunaXGBoost | 0,286019 | 0,25 | 1 |
| OptunaXGBoost | 0,088097 | 0,25 | 0 |

## Conclusao

O projeto mostra que existe sinal preditivo real na base, mas tambem mostra uma
limitacao importante: as variaveis disponiveis sao majoritariamente
demograficas, territoriais e cadastrais. Mesmo apos KNN, Arvore de Decisao,
feature engineering, calibracao, reamostragem, ensembles, boosting, Optuna,
stacking e selecao de atributos, os melhores resultados estabilizaram perto de:

```text
Average Precision ~= 0,20
ROC-AUC ~= 0,70
F1 ~= 0,28
```

Assim, o modelo e mais adequado como ferramenta de ranking e triagem de risco do
que como decisor automatico de concessao ou recusa de credito.
