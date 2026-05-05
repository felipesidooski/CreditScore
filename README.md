# Credit Score - PUCPR

Projeto de Modelagem Avancada de Dados para predicao de inadimplencia em uma
base de credit scoring.

O objetivo e estimar se um cliente quitara sua divida integralmente
(`TARGET=0`) ou se entrara em inadimplencia (`TARGET=1`). A entrega compara os
modelos obrigatorios da rubrica, KNN e Arvore de Decisao, e tambem apresenta
experimentos avancados com XGBoost, LightGBM, stacking, selecao de atributos,
Optuna, calibracao e reamostragem.

## Como usar no Colab

```bash
git clone <URL_DO_REPOSITORIO_CREDITSCORE>
cd CreditScore
pip install -r requirements.txt
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
├── notebooks/
│   └── credit_score.ipynb
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

## Modelos implementados

Os experimentos historicos foram separados por familia de metodo em:

```text
legacy/
```

O script completo original foi preservado apenas como referencia em:

```text
legacy/archive/credit_score_experiments_full.py
```

As classes finais estao nos modulos:

```text
src/creditscore/training.py
src/creditscore/inference.py
```

### `ChampionModelTrainer`

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

Metricas principais:

- `Average Precision (AP)`: qualidade do ranking da classe inadimplente.
- `Precision`: dos clientes marcados como risco, quantos realmente eram
  inadimplentes.
- `Recall`: dos inadimplentes reais, quantos o modelo encontrou.
- `F1`: equilibrio entre precision e recall.
- `ROC-AUC`: capacidade geral de separacao entre classes.

Baseline:

| Modelo | Accuracy | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0,9044 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,0956 |

Evolucao dos principais experimentos:

| Modelo | Threshold | Accuracy | Precision_1 | Recall_1 | F1_1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN original | 0,50 | 0,9025 | 0,2167 | 0,0074 | 0,0143 | 0,6017 | 0,1393 |
| DecisionTree original | 0,50 | 0,5698 | 0,1381 | 0,6674 | 0,2288 | 0,6545 | 0,1546 |
| AdvancedDecisionTree calibrated | 0,10 | 0,6161 | 0,1478 | 0,6322 | 0,2396 | 0,6740 | 0,1775 |
| AdvancedXGBoost | 0,40 | 0,7988 | 0,2110 | 0,4030 | 0,2770 | 0,6970 | 0,1985 |
| OptunaXGBoost | 0,25 | 0,7947 | 0,2094 | 0,4132 | 0,2780 | 0,6990 | 0,2006 |
| ImportanceXGBoost_top50 | 0,35 | 0,7229 | 0,1800 | 0,5335 | 0,2692 | 0,6967 | 0,2020 |

Arquivos com resultados completos:

```text
reports/tables/comparison_all_experiments.csv
reports/tables/importance_selection_holdout_metrics.csv
reports/tables/optuna_xgboost_holdout_metrics.csv
reports/tables/benchmark_ensemble_holdout_metrics.csv
reports/tables/resampling_holdout_metrics.csv
reports/tables/native_boosting_holdout_metrics.csv
reports/tables/stacking_holdout_metrics.csv
```

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
- parametros escolhidos por busca bayesiana com Optuna;
- boa alternativa quando o foco e separacao global entre adimplentes e
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
