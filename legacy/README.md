# Legacy experiments

Este diretorio organiza os experimentos historicos por familia de metodo.

O arquivo completo original foi preservado em:

```text
legacy/archive/credit_score_experiments_full.py
```

Os modulos da raiz de `legacy/` expõem grupos independentes de funcoes:

| Arquivo | Conteudo |
|---|---|
| `common.py` | configuracao e utilitarios de I/O |
| `eda.py` | apresentacao da base e AED |
| `preprocessing.py` | limpeza deterministica e feature engineering |
| `pipelines.py` | ColumnTransformer, pipelines e target encoder |
| `evaluation.py` | metricas, thresholds e calibracao |
| `original_models.py` | KNN e Arvore de Decisao originais |
| `enhanced_models.py` | trilha enhanced |
| `advanced_models.py` | trilha advanced |
| `ensemble_benchmarks.py` | RandomForest, ExtraTrees, boosting e native boosting |
| `resampling.py` | SMOTE e NearMiss |
| `optuna_xgboost.py` | busca Optuna para XGBoost |
| `feature_selection.py` | ranking SHAP/importancia e top-N features |
| `stacking.py` | StackingClassifier |
| `target_encoding.py` | detecção e teste de target encoding |
| `decision_validation.py` | validacao empirica das decisoes de limpeza/modelagem |

