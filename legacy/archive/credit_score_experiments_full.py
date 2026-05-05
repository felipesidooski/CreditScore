"""Credit Score - AED, cleanup, validacao de decisoes e modelagem.

Este script espelha o conteudo planejado para o notebook `credit_score.ipynb`,
mas em formato `.py` para facilitar testes locais, depuracao e execucao por
etapas antes de levar a versao final ao Colab.

Fluxo geral:

1. Carregar `train.csv` bruto.
2. Apresentar a base como recebida, incluindo os codigos sentinela.
3. Converter sentinelas (-99, -9998, -9999) para NaN.
4. Criar flags deterministicas de ausencia.
5. Remover colunas proibidas/legalmente sensiveis e colunas com ausencia extrema.
6. Fazer split estratificado antes de qualquer imputacao/normalizacao.
7. Montar `ColumnTransformer` dentro de `Pipeline`.
8. Comparar KNN e Arvore de Decisao com `GridSearchCV`.
9. Avaliar no holdout e comparar contra `DummyClassifier`.

Decisao central:

Nao usamos `cleaned_train.csv` para modelagem. Ele ja passou por imputacao e
normalizacao calculadas sobre toda a base, o que cria risco de data leakage.
Aqui, toda estatistica aprendida dos dados (mediana, escala, etc.) fica dentro
do pipeline e e ajustada apenas no treino de cada fold.

Como executar:

    python credit_score.py --all --fast

Etapas individuais:

    python credit_score.py --eda
    python credit_score.py --validate-decisions --fast
    python credit_score.py --model --fast
    python credit_score.py --model --enhanced-model --fast
    python credit_score.py --model --enhanced-model --advanced-model --fast
    python credit_score.py --advanced-model --benchmark-ensembles --fast
    python credit_score.py --advanced-model --resampling-models --fast
    python credit_score.py --advanced-model --native-boosting --optuna-xgboost --fast
    python credit_score.py --advanced-model --advanced-extras --fast

Para a execucao final, remova `--fast`.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _import_dependencies() -> None:
    """Importa dependencias com mensagem amigavel quando a venv esta vazia.

    A venv do projeto foi criada na raiz de `CREDITSCORE`, mas pode ainda nao
    ter as bibliotecas instaladas. Em vez de falhar com um stack trace pouco
    explicativo, mostramos o comando exato para instalar o necessario.
    """

    missing = []
    for module_name in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"]:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        print("Dependencias ausentes:", ", ".join(missing))
        print()
        print("Instale na venv do projeto com:")
        print("  ../bin/python -m pip install numpy pandas matplotlib seaborn scikit-learn")
        print()
        print("Extras opcionais para experimentos avancados:")
        print("  ../bin/python -m pip install imbalanced-learn lightgbm xgboost optuna shap")
        print()
        print("Depois execute novamente:")
        print("  ../bin/python credit_score.py --all --fast")
        raise SystemExit(1)


_import_dependencies()

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
sns.set_theme(style="whitegrid", palette="Set2")


@dataclass(frozen=True)
class Config:
    """Configuracao central do experimento.

    Manter esses parametros em uma classe evita valores soltos pelo codigo e
    facilita alterar o experimento sem quebrar a reprodutibilidade.
    """

    random_state: int = 42
    target_col: str = "TARGET"
    sentinels: tuple[int, ...] = (-99, -9998, -9999)
    test_size: float = 0.20
    cv_folds: int = 5
    scoring: str = "average_precision"


CONFIG = Config()


LEGAL_SENSITIVE_COLS = ["ORIENTACAO_SEXUAL", "RELIGIAO"]

CASA_COLS = [
    "QTDPESSOASCASA",
    "MENORRENDACASA",
    "MAIORRENDACASA",
    "SOMARENDACASA",
    "MEDIARENDACASA",
    "MAIORIDADECASA",
    "MENORIDADECASA",
    "MEDIAIDADECASA",
    "INDICMENORDEIDADE",
    "COBRANCABAIXOCASA",
    "COBRANCAMEDIOCASA",
    "COBRANCAALTACASA",
    "SEGMENTACAOFINBAIXACASA",
    "SEGMENTACAOFINMEDIACASA",
    "SEGMENTACAOALTACASA",
    "BOLSAFAMILIACASA",
    "FUNCIONARIOPUBLICOCASA",
]


ATTRIBUTE_DESCRIPTIONS = [
    ("TARGET", "Alvo: 0 indica quitacao integral; 1 indica inadimplencia."),
    ("HS_CPF", "Identificador do cliente; removido da modelagem."),
    ("TEMPOCPF", "Tempo associado ao CPF/cadastro, possivel estabilidade cadastral."),
    ("DISTCENTROCIDADE", "Distancia ate o centro da cidade."),
    ("DISTZONARISCO", "Distancia ate zona de risco."),
    ("QTDENDERECO", "Quantidade de enderecos registrados."),
    ("QTDCELULAR", "Quantidade de celulares vinculados ao cliente."),
    ("QTDFONEFIXO", "Quantidade de telefones fixos vinculados."),
    ("ESTIMATIVARENDA", "Estimativa de renda individual."),
    ("MEDIARENDACEP", "Renda media estimada da regiao/CEP."),
    ("IDHMUNICIPIO", "Indicador de desenvolvimento humano do municipio."),
    ("PIBMUNICIPIO", "Indicador economico do municipio."),
    ("QTDPESSOASCASA", "Quantidade estimada de pessoas na residencia."),
    ("MEDIARENDACASA", "Renda media estimada da residencia."),
    ("ANOSULTIMADECLARACAO", "Anos desde a ultima declaracao fiscal identificada."),
    ("ORIENTACAO_SEXUAL", "Variavel sensivel; descrita na AED e removida do modelo."),
    ("RELIGIAO", "Variavel sensivel; descrita na AED e removida do modelo."),
]


def parse_args() -> argparse.Namespace:
    """Define a interface de linha de comando.

    A ideia e permitir testes incrementais. Primeiro voce roda `--eda`, depois
    valida as discordancias com `--validate-decisions`, e so entao executa
    `--model`, que e a parte mais cara.
    """

    parser = argparse.ArgumentParser(description="Projeto Credit Score")
    parser.add_argument("--data-path", default="train.csv", help="Caminho do train.csv.")
    parser.add_argument("--output-dir", default="outputs", help="Diretorio dos relatorios.")
    parser.add_argument("--eda", action="store_true", help="Executa apresentacao e AED textual.")
    parser.add_argument(
        "--validate-decisions",
        action="store_true",
        help="Compara decisoes discutidas: scaler, ANOSULTIMADECLARACAO e flags.",
    )
    parser.add_argument("--model", action="store_true", help="Executa GridSearchCV e holdout.")
    parser.add_argument(
        "--enhanced-model",
        action="store_true",
        help=(
            "Executa uma trilha experimental adicional com feature engineering, "
            "grid expandido e threshold escolhido em validacao."
        ),
    )
    parser.add_argument(
        "--advanced-model",
        action="store_true",
        help=(
            "Executa uma terceira trilha com features de razao/zero, grid refinado, "
            "threshold por F1/F2/custo e calibracao."
        ),
    )
    parser.add_argument(
        "--benchmark-ensembles",
        action="store_true",
        help=(
            "Inclui benchmarks complementares com RandomForest, ExtraTrees, "
            "HistGradientBoosting e boosting opcional (LightGBM/XGBoost quando "
            "instalados). Nao faz parte da comparacao principal KNN vs DT."
        ),
    )
    parser.add_argument(
        "--resampling-models",
        action="store_true",
        help=(
            "Inclui experimentos opcionais com reamostragem via imbalanced-learn "
            "(SMOTE e NearMiss) dentro do pipeline/CV. Nao substitui os modelos "
            "obrigatorios."
        ),
    )
    parser.add_argument(
        "--native-boosting",
        action="store_true",
        help=(
            "Testa XGBoost/LightGBM preservando NaN nativo, sem SimpleImputer e "
            "sem scaler. Experimento opcional para comparar contra o pipeline "
            "sklearn padrao."
        ),
    )
    parser.add_argument(
        "--optuna-xgboost",
        action="store_true",
        help=(
            "Executa busca bayesiana com Optuna para XGBoost nativo. Requer "
            "optuna e xgboost instalados."
        ),
    )
    parser.add_argument(
        "--importance-selection",
        action="store_true",
        help=(
            "Testa selecao de features por importancia do XGBoost, incluindo "
            "top-N features. SHAP e usado quando estiver instalado; caso "
            "contrario usa feature_importances_."
        ),
    )
    parser.add_argument(
        "--stacking-models",
        action="store_true",
        help=(
            "Testa StackingClassifier com modelos fortes como XGBoost, LightGBM "
            "e HistGradientBoosting quando disponiveis."
        ),
    )
    parser.add_argument(
        "--target-encoding",
        action="store_true",
        help=(
            "Procura colunas que parecam codigos categoricos numericos e testa "
            "target encoding dentro do pipeline/CV quando houver candidatas."
        ),
    )
    parser.add_argument(
        "--advanced-extras",
        action="store_true",
        help=(
            "Atalho para executar native boosting, Optuna, importance selection, "
            "stacking e target encoding."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Executa todas as etapas.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduz amostras e grids para testes rapidos.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """Resolve caminhos relativos a partir da pasta do script."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def ensure_output_dir(path_value: str) -> Path:
    """Cria o diretorio de saida e subpastas usadas pelo script."""

    output_dir = resolve_path(path_value)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_table(df: pd.DataFrame, output_dir: Path, name: str) -> Path:
    """Salva tabelas em CSV para facilitar inspecao fora do notebook."""

    path = output_dir / "tables" / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_plot(output_dir: Path, name: str) -> Path:
    """Salva o grafico atual e fecha a figura para economizar memoria."""

    path = output_dir / "plots" / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Carrega `train.csv` sem transformacao.

    Esta funcao propositalmente nao substitui sentinelas nem altera tipos. A
    primeira apresentacao da base precisa mostrar o arquivo como ele veio.
    """

    df = pd.read_csv(data_path)
    print(f"Arquivo carregado: {data_path.resolve()}")
    print(f"Shape bruto: {df.shape[0]:,} linhas x {df.shape[1]:,} colunas")
    return df


def build_missing_report(df_raw: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Calcula ausencia real considerando codigos sentinela.

    O CSV nao possui NaN explicito, mas possui sentinelas. Para a rubrica de
    percentual de dados faltantes, os sentinelas precisam ser contabilizados.
    """

    sentinel_counts = df_raw.isin(config.sentinels).sum()
    report = pd.DataFrame(
        {
            "coluna": df_raw.columns,
            "nan_explicito_pct": (df_raw.isna().mean() * 100).round(2).values,
            "sentinelas_pct": (sentinel_counts / len(df_raw) * 100).round(2).values,
            "sentinelas_qtd": sentinel_counts.values,
            "dtype": df_raw.dtypes.astype(str).values,
        }
    )
    return report.sort_values(
        ["sentinelas_pct", "sentinelas_qtd"],
        ascending=False,
    ).reset_index(drop=True)


def present_dataset(df_raw: pd.DataFrame, output_dir: Path, config: Config) -> None:
    """Apresenta shape, TARGET, tipos, missing e dicionario resumido.

    Esta etapa cobre a parte de apresentacao do conjunto de dados exigida pela
    rubrica. As tabelas sao salvas em CSV para facilitar revisao.
    """

    target_counts = df_raw[config.target_col].value_counts(dropna=False).sort_index()
    target_distribution = pd.DataFrame(
        {
            "classe": target_counts.index.astype(str),
            "quantidade": target_counts.values,
            "percentual": (target_counts.values / len(df_raw) * 100).round(2),
        }
    )
    dtype_counts = (
        df_raw.dtypes.value_counts()
        .rename_axis("dtype")
        .reset_index(name="quantidade")
    )
    missing_report = build_missing_report(df_raw, config)
    attr_desc = pd.DataFrame(ATTRIBUTE_DESCRIPTIONS, columns=["atributo", "descricao"])

    save_table(target_distribution, output_dir, "target_distribution")
    save_table(dtype_counts, output_dir, "dtype_counts")
    save_table(missing_report, output_dir, "missing_sentinel_report")
    save_table(attr_desc, output_dir, "attribute_descriptions")

    print("\n=== Apresentacao do dataset ===")
    print(f"Instancias: {len(df_raw):,}")
    print(f"Atributos totais: {df_raw.shape[1]:,}")
    print("\nDistribuicao do TARGET:")
    print(target_distribution.to_string(index=False))
    print("\nTop 20 colunas com sentinelas:")
    print(missing_report.query("sentinelas_qtd > 0").head(20).to_string(index=False))
    print("\nDescricoes de atributos salvas em outputs/tables/attribute_descriptions.csv")


def make_eda_frame(df_raw: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Cria copia para AED com sentinelas convertidos para NaN.

    Essa conversao e deterministica: nao aprende mediana, media, desvio padrao
    ou qualquer estatistica. Ela apenas impede que -9999 distorca graficos e
    estatisticas descritivas.
    """

    df_eda = df_raw.replace(list(config.sentinels), np.nan).copy()
    df_eda[config.target_col] = df_eda[config.target_col].astype(int)
    existing_casa_cols = [col for col in CASA_COLS if col in df_eda.columns]
    df_eda["GRUPO_CASA_AUSENTE_EDA"] = (
        df_eda[existing_casa_cols].isna().any(axis=1).astype(int)
    )
    df_eda["DECLARACAO_AUSENTE_EDA"] = (
        df_eda["ANOSULTIMADECLARACAO"].isna().astype(int)
    )
    return df_eda


def numeric_summary_by_target(
    df: pd.DataFrame,
    col: str,
    config: Config,
) -> pd.DataFrame:
    """Resume uma variavel numerica por classe do TARGET."""

    valid = df[[col, config.target_col]].dropna()
    if valid.empty:
        return pd.DataFrame()
    return valid.groupby(config.target_col)[col].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()


def target_rate_by_column(
    df: pd.DataFrame,
    col: str,
    config: Config,
) -> pd.DataFrame:
    """Calcula taxa de inadimplencia por categoria/valor discreto."""

    result = (
        df.groupby(col, dropna=False)[config.target_col]
        .agg(quantidade="count", taxa_inadimplencia="mean")
        .reset_index()
    )
    result["taxa_inadimplencia_pct"] = (
        result["taxa_inadimplencia"] * 100
    ).round(2)
    return result


def plot_numeric_by_target(
    df: pd.DataFrame,
    col: str,
    output_dir: Path,
    config: Config,
    name: str,
    clip_quantile: float = 0.99,
) -> None:
    """Gera histograma e boxplot por TARGET.

    O clipping aqui e apenas visual: nao altera os dados usados no modelo. Isso
    deixa graficos legiveis quando ha caudas longas.
    """

    valid = df[[col, config.target_col]].dropna().copy()
    if valid.empty:
        return
    upper = valid[col].quantile(clip_quantile)
    valid_plot = valid[valid[col] <= upper]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(
        data=valid_plot,
        x=col,
        hue=config.target_col,
        stat="density",
        common_norm=False,
        bins=30,
        ax=axes[0],
    )
    axes[0].set_title(f"Distribuicao por TARGET - {col}")
    sns.boxplot(data=valid_plot, x=config.target_col, y=col, ax=axes[1])
    axes[1].set_title(f"Boxplot por TARGET - {col}")
    save_plot(output_dir, name)


def plot_target_rate(
    summary: pd.DataFrame,
    col: str,
    output_dir: Path,
    name: str,
) -> None:
    """Gera barplot de taxa de inadimplencia por uma coluna discreta."""

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=summary, x=col, y="taxa_inadimplencia", ax=ax)
    ax.set_title(f"Taxa de inadimplencia por {col}")
    ax.set_ylabel("Taxa de inadimplencia")
    save_plot(output_dir, name)


def run_univariate_eda(
    df_eda: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> None:
    """Executa as 15 analises univariadas exigidas.

    Cada analise e registrada com questao/hipotese em texto e uma tabela ou
    grafico salvo. No notebook, esses textos podem ser usados como celulas
    Markdown.
    """

    analyses = []

    # 1 TARGET
    target_summary = target_rate_by_column(df_eda, config.target_col, config)
    save_table(target_summary, output_dir, "u01_target_distribution")
    analyses.append(
        (
            "U01 TARGET",
            "Qual e o desbalanceamento entre adimplentes e inadimplentes?",
            "A classe inadimplente deve ser minoritaria; accuracy nao decide o melhor modelo.",
        )
    )

    # 2 Missing/sentinelas ja foi reportado na apresentacao.
    analyses.append(
        (
            "U02 Sentinelas",
            "Quais colunas concentram codigos de ausencia?",
            "Colunas com muita ausencia precisam de remocao, flag ou imputacao dentro do pipeline.",
        )
    )

    numeric_analyses = [
        ("U03 TEMPOCPF", "TEMPOCPF", "Clientes com maior tempo de CPF inadimplem menos?"),
        ("U04 ESTIMATIVARENDA", "ESTIMATIVARENDA", "A renda estimada difere por TARGET?"),
        ("U05 DISTZONARISCO", "DISTZONARISCO", "Distancia a zona de risco diferencia os grupos?"),
        ("U06 DISTCENTROCIDADE", "DISTCENTROCIDADE", "Distancia ao centro esta associada ao risco?"),
        ("U07 QTDCELULAR", "QTDCELULAR", "Quantidade de celulares diferencia os grupos?"),
        ("U08 QTDENDERECO", "QTDENDERECO", "Quantidade de enderecos sugere instabilidade?"),
        ("U11 ANOSULTIMADECLARACAO", "ANOSULTIMADECLARACAO", "Declaracao fiscal carrega sinal?"),
        ("U13 MEDIARENDACASA", "MEDIARENDACASA", "Renda media da casa difere por TARGET?"),
        ("U14 IDHMUNICIPIO", "IDHMUNICIPIO", "IDH municipal diferencia os grupos?"),
    ]

    for code_name, col, question in numeric_analyses:
        summary = numeric_summary_by_target(df_eda, col, config)
        save_table(summary, output_dir, code_name.lower().replace(" ", "_"))
        plot_numeric_by_target(
            df_eda,
            col,
            output_dir,
            config,
            code_name.lower().replace(" ", "_"),
            clip_quantile=1.0 if col in {"QTDCELULAR", "QTDENDERECO"} else 0.99,
        )
        analyses.append((code_name, question, "Variavel mantida inicialmente e escalada no pipeline."))

    binary_analyses = [
        ("U09 FUNCIONARIOPUBLICO", "FUNCIONARIOPUBLICO", "Funcionario publico altera a taxa de inadimplencia?"),
        ("U10 BOLSAFAMILIA", "BOLSAFAMILIA", "Bolsa Familia esta associada a inadimplencia?"),
        ("U12 GRUPO_CASA_AUSENTE", "GRUPO_CASA_AUSENTE_EDA", "Ausencia de informacoes CASA muda a taxa?"),
        ("U11B DECLARACAO_AUSENTE", "DECLARACAO_AUSENTE_EDA", "Ausencia de declaracao muda a taxa?"),
    ]

    for code_name, col, question in binary_analyses:
        summary = target_rate_by_column(df_eda, col, config)
        save_table(summary, output_dir, code_name.lower().replace(" ", "_"))
        plot_target_rate(summary, col, output_dir, code_name.lower().replace(" ", "_"))
        analyses.append((code_name, question, "A taxa por grupo orienta a decisao de criar flags."))

    # 15 Variaveis sensiveis.
    for sensitive_col in LEGAL_SENSITIVE_COLS:
        if sensitive_col in df_eda.columns:
            summary = target_rate_by_column(df_eda, sensitive_col, config)
            save_table(summary, output_dir, f"u15_{sensitive_col.lower()}")
    analyses.append(
        (
            "U15 Variaveis sensiveis",
            "Como ORIENTACAO_SEXUAL e RELIGIAO aparecem na base?",
            "Sao descritas na AED, mas removidas antes da modelagem.",
        )
    )

    analysis_text = pd.DataFrame(
        analyses,
        columns=["analise", "questao_hipotese", "discussao"],
    )
    save_table(analysis_text, output_dir, "univariate_analysis_questions")
    print("\n=== AED univariada ===")
    print("Analises univariadas salvas em outputs/tables e outputs/plots.")


def run_multivariate_eda(
    df_eda: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> None:
    """Executa 5 analises multivariadas.

    O objetivo e mostrar relacoes entre conjuntos de variaveis, nao apenas
    distribuicoes isoladas. As tabelas geradas apoiam a discussao do relatorio.
    """

    selected_corr_cols = [
        "TEMPOCPF",
        "ESTIMATIVARENDA",
        "MEDIARENDACEP",
        "MEDIARENDACASA",
        "MAIORRENDACASA",
        "SOMARENDACASA",
        "IDHMUNICIPIO",
        "PIBMUNICIPIO",
        "DISTCENTROCIDADE",
        "DISTZONARISCO",
        config.target_col,
    ]
    selected_corr_cols = [col for col in selected_corr_cols if col in df_eda.columns]
    corr = df_eda[selected_corr_cols].corr(numeric_only=True)
    corr.to_csv(output_dir / "tables" / "m01_correlation_selected.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlacao - variaveis selecionadas")
    save_plot(output_dir, "m01_correlation_selected")

    renda_decis = df_eda[["ESTIMATIVARENDA", config.target_col]].dropna().copy()
    renda_decis["decil_renda"] = pd.qcut(
        renda_decis["ESTIMATIVARENDA"],
        q=10,
        duplicates="drop",
    )
    renda_summary = (
        renda_decis.groupby("decil_renda", observed=False)[config.target_col]
        .agg(quantidade="count", taxa_inadimplencia="mean")
        .reset_index()
    )
    save_table(renda_summary, output_dir, "m03_income_deciles")

    missing_combo = (
        df_eda.groupby(["GRUPO_CASA_AUSENTE_EDA", "DECLARACAO_AUSENTE_EDA"])[
            config.target_col
        ]
        .agg(quantidade="count", taxa_inadimplencia="mean")
        .reset_index()
    )
    save_table(missing_combo, output_dir, "m04_missingness_combo")

    numeric_for_corr = df_eda.select_dtypes(include="number").drop(
        columns=[config.target_col],
        errors="ignore",
    )
    abs_corr = numeric_for_corr.corr().abs()
    pairs = []
    cols = abs_corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = abs_corr.loc[col_a, col_b]
            if pd.notna(value) and value >= 0.85:
                pairs.append((col_a, col_b, value))

    high_corr_pairs = pd.DataFrame(
        pairs,
        columns=["feature_a", "feature_b", "abs_corr"],
    ).sort_values("abs_corr", ascending=False)
    save_table(high_corr_pairs, output_dir, "m05_high_corr_pairs")

    questions = pd.DataFrame(
        [
            (
                "M01 Correlacao",
                "Quais variaveis financeiras/territoriais sao redundantes?",
                "Correlacao alta orienta discussao, mas nao implica causalidade.",
            ),
            (
                "M02 Renda x IDH",
                "A relacao renda/IDH muda por TARGET?",
                "Interacao entre renda individual e contexto regional pode ser relevante.",
            ),
            (
                "M03 Decis de renda",
                "A inadimplencia varia por faixa de renda?",
                "Decis reduzem ruido e mostram tendencia agregada.",
            ),
            (
                "M04 Ausencias combinadas",
                "Ausencias simultaneas alteram inadimplencia?",
                "Justifica flags deterministicas de ausencia.",
            ),
            (
                "M05 Alta correlacao",
                "Ha pares com correlacao absoluta >= 0,85?",
                "Ajuda a discutir redundancia e possivel versao reduzida.",
            ),
        ],
        columns=["analise", "questao_hipotese", "discussao"],
    )
    save_table(questions, output_dir, "multivariate_analysis_questions")
    print("\n=== AED multivariada ===")
    print("Analises multivariadas salvas em outputs/tables e outputs/plots.")


def deterministic_cleanup(
    df_raw: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aplica apenas limpeza deterministica antes do split.

    O que pode acontecer antes do split:
    - mapear sentinelas para NaN;
    - criar flags a partir de NaN/nao-NaN;
    - remover colunas proibidas ou com ausencia extrema.

    O que nao pode acontecer antes do split:
    - imputar mediana;
    - calcular escala;
    - fazer clipping por quantis;
    - selecionar features por desempenho.
    """

    df_model = df_raw.copy().replace(list(config.sentinels), np.nan)
    df_model[config.target_col] = df_model[config.target_col].astype(int)

    existing_casa_cols = [col for col in CASA_COLS if col in df_model.columns]
    casa_missing_matrix = df_model[existing_casa_cols].isna()
    all_missing_equals_any_missing = casa_missing_matrix.all(axis=1).equals(
        casa_missing_matrix.any(axis=1)
    )
    unique_casa_missing_patterns = casa_missing_matrix.drop_duplicates().shape[0]

    if all_missing_equals_any_missing:
        df_model["GRUPO_CASA_AUSENTE"] = (
            df_model[existing_casa_cols[0]].isna().astype(int)
        )
        casa_flag_decision = "flag_unica"
    else:
        df_model["GRUPO_CASA_AUSENTE"] = (
            df_model[existing_casa_cols].isna().any(axis=1).astype(int)
        )
        casa_flag_decision = "flag_agregada_any"

    df_model["DECLARACAO_AUSENTE"] = (
        df_model["ANOSULTIMADECLARACAO"].isna().astype(int)
    )

    cols_drop_legal = ["HS_CPF", *LEGAL_SENSITIVE_COLS]
    cols_drop_extreme_missing = [
        "ANOSULTIMARESTITUICAO",
        "ANOSULTIMADECLARACAOPAGAR",
    ]
    cols_to_drop = [
        col for col in [*cols_drop_legal, *cols_drop_extreme_missing]
        if col in df_model.columns
    ]
    df_model = df_model.drop(columns=cols_to_drop)

    metadata = {
        "existing_casa_cols": existing_casa_cols,
        "all_missing_equals_any_missing": all_missing_equals_any_missing,
        "unique_casa_missing_patterns": unique_casa_missing_patterns,
        "casa_flag_decision": casa_flag_decision,
        "removed_columns": cols_to_drop,
        "shape_after_cleanup": df_model.shape,
    }
    return df_model, metadata


LOG_FEATURE_COLS = [
    "ESTIMATIVARENDA",
    "MEDIARENDACEP",
    "PIBMUNICIPIO",
    "DISTCENTROCIDADE",
    "DISTZONARISCO",
    "MEDIARENDACASA",
    "MAIORRENDACASA",
    "SOMARENDACASA",
]


def add_enhanced_features(
    df_model: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adiciona features deterministicas para a trilha enhanced.

    Esta funcao preserva a base original de modelagem e cria uma copia com
    novas variaveis. Todas as transformacoes abaixo sao permitidas antes do
    split porque nao aprendem estatisticas globais:

    - contagem/proporcao de campos ausentes por registro;
    - flags agregadas por blocos de variaveis;
    - transformacoes logaritmicas `log1p` sobre valores existentes.

    O objetivo e testar se sinais de ausencia e caudas longas ajudam KNN/Arvore
    sem perder o comparativo com o pipeline original.
    """

    df = df_model.copy()
    feature_cols = [col for col in df.columns if col != config.target_col]

    df["QTD_CAMPOS_AUSENTES"] = df[feature_cols].isna().sum(axis=1)
    df["PCT_CAMPOS_AUSENTES"] = df[feature_cols].isna().mean(axis=1)

    existing_casa_cols = [col for col in CASA_COLS if col in df.columns]
    if existing_casa_cols:
        casa_missing = df[existing_casa_cols].isna()
        df["CASA_QTD_AUSENTES"] = casa_missing.sum(axis=1)
        df["CASA_TOTALMENTE_AUSENTE"] = casa_missing.all(axis=1).astype(int)
        df["CASA_PARCIALMENTE_AUSENTE"] = (
            casa_missing.any(axis=1) & ~casa_missing.all(axis=1)
        ).astype(int)

    cep_cols = [
        col for col in df.columns
        if col.endswith("CEP") and col != config.target_col
    ]
    municipio_cols = [
        col for col in df.columns
        if col.endswith("MUNICIPIO") and col != config.target_col
    ]

    if cep_cols:
        cep_missing = df[cep_cols].isna()
        df["CEP_QTD_AUSENTES"] = cep_missing.sum(axis=1)
        df["CEP_AUSENTE"] = cep_missing.any(axis=1).astype(int)

    if municipio_cols:
        municipio_missing = df[municipio_cols].isna()
        df["MUNICIPIO_QTD_AUSENTES"] = municipio_missing.sum(axis=1)
        df["MUNICIPIO_AUSENTE"] = municipio_missing.any(axis=1).astype(int)

    created_log_cols = []
    for col in LOG_FEATURE_COLS:
        if col not in df.columns:
            continue
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        # log1p exige valores >= -1. Como estas variaveis representam renda,
        # distancia ou agregados economicos, valores negativos restantes nao
        # teriam interpretacao natural; por seguranca, limitamos em zero.
        new_col = f"LOG_{col}"
        df[new_col] = np.log1p(numeric_series.clip(lower=0))
        created_log_cols.append(new_col)

    metadata = pd.DataFrame(
        [
            ("features_originais", len(feature_cols)),
            ("features_enhanced", df.shape[1] - 1),
            ("novas_features", df.shape[1] - df_model.shape[1]),
            ("casa_cols_usadas", len(existing_casa_cols)),
            ("cep_cols_usadas", len(cep_cols)),
            ("municipio_cols_usadas", len(municipio_cols)),
            ("log_features_criadas", created_log_cols),
        ],
        columns=["chave", "valor"],
    )
    save_table(metadata, output_dir, "enhanced_feature_metadata")
    return df, metadata


ZERO_FLAG_COLS = [
    "QTDEMAIL",
    "QTDCELULAR",
    "QTDFONEFIXO",
    "QTDENDERECO",
    "QTDDECLARACAOISENTA",
    "QTDDECLARACAO10",
    "QTDDECLARACAOREST10",
    "QTDDECLARACAOPAGAR10",
    "RESTITUICAOAGENCIAALTARENDA",
    "BOLSAFAMILIA",
    "FUNCIONARIOPUBLICO",
    "SOCIOEMPRESA",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calcula razoes evitando divisao por zero e infinitos."""

    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_advanced_features(
    df_model: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adiciona features da terceira trilha experimental.

    Esta trilha parte da base enhanced e adiciona:

    - flags de zero para variaveis de contagem ou presenca;
    - razoes entre renda individual, renda regional e renda domiciliar;
    - razoes territoriais e economicas simples;
    - agregados de veiculos no municipio.

    Todas sao transformacoes linha-a-linha. Portanto, nao causam leakage.
    """

    df, enhanced_metadata = add_enhanced_features(df_model, output_dir, config)
    created_zero_flags = []
    created_ratio_cols = []

    for col in ZERO_FLAG_COLS:
        if col not in df.columns:
            continue
        new_col = f"{col}_ZERO"
        df[new_col] = (pd.to_numeric(df[col], errors="coerce") == 0).astype(int)
        created_zero_flags.append(new_col)

    def add_ratio(new_col: str, numerator_col: str, denominator_col: str) -> None:
        if numerator_col not in df.columns or denominator_col not in df.columns:
            return
        numerator = pd.to_numeric(df[numerator_col], errors="coerce")
        denominator = pd.to_numeric(df[denominator_col], errors="coerce")
        df[new_col] = safe_ratio(numerator, denominator)
        created_ratio_cols.append(new_col)

    add_ratio("RAZAO_RENDA_ESTIMADA_CEP", "ESTIMATIVARENDA", "MEDIARENDACEP")
    add_ratio("RAZAO_RENDA_ESTIMADA_CASA", "ESTIMATIVARENDA", "MEDIARENDACASA")
    add_ratio("RAZAO_RENDA_CASA_CEP", "MEDIARENDACASA", "MEDIARENDACEP")
    add_ratio("RENDA_TOTAL_CASA_POR_PESSOA", "SOMARENDACASA", "QTDPESSOASCASA")
    add_ratio("RAZAO_DIST_RISCO_CENTRO", "DISTZONARISCO", "DISTCENTROCIDADE")
    add_ratio("RAZAO_RENDA_CEP_PIB", "MEDIARENDACEP", "PIBMUNICIPIO")
    add_ratio("RAZAO_RENDA_ESTIMADA_IDH", "ESTIMATIVARENDA", "IDHMUNICIPIO")

    vehicle_cols = [
        "QTDUTILITARIOMUNICIPIO",
        "QTDAUTOMOVELMUNICIPIO",
        "QTDCAMINHAOMUNICIPIO",
        "QTDCAMINHONETEMUNICIPIO",
        "QTDMOTOMUNICIPIO",
    ]
    existing_vehicle_cols = [col for col in vehicle_cols if col in df.columns]
    if existing_vehicle_cols:
        df["TOTAL_VEICULOS_MUNICIPIO"] = (
            df[existing_vehicle_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        )
        created_ratio_cols.append("TOTAL_VEICULOS_MUNICIPIO")
        add_ratio("RAZAO_RENDA_ESTIMADA_VEICULOS", "ESTIMATIVARENDA", "TOTAL_VEICULOS_MUNICIPIO")

    metadata = pd.DataFrame(
        [
            ("features_enhanced_base", int(enhanced_metadata.loc[enhanced_metadata["chave"] == "features_enhanced", "valor"].iloc[0])),
            ("features_advanced", df.shape[1] - 1),
            ("novas_features_advanced", df.shape[1] - df_model.shape[1]),
            ("zero_flags_criadas", created_zero_flags),
            ("ratio_features_criadas", created_ratio_cols),
        ],
        columns=["chave", "valor"],
    )
    save_table(metadata, output_dir, "advanced_feature_metadata")
    return df, metadata


def make_ohe() -> OneHotEncoder:
    """Cria OneHotEncoder compativel com versoes novas e antigas do sklearn."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scaler,
) -> ColumnTransformer:
    """Monta o ColumnTransformer exigido pela rubrica.

    Numericas:
    - `SimpleImputer(strategy="median")`, pois a mediana e robusta a assimetria;
    - scaler definido pelo experimento.

    Categoricas:
    - imputacao constante;
    - OneHotEncoder.

    Depois da remocao legal de ORIENTACAO_SEXUAL e RELIGIAO, e esperado que
    `categorical_cols` fique vazio. Mesmo assim, mantemos o bloco para documentar
    a estrutura exigida e para robustez em bases futuras.
    """

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            ("ohe", make_ohe()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Target encoding simples e seguro para uso dentro de Pipeline/CV.

    O transformer aprende a media do TARGET por categoria apenas no `fit`.
    Quando esta dentro de um Pipeline usado pelo GridSearchCV, cada fold aprende
    essas medias somente no treino daquele fold. Isso evita o vazamento classico
    do target encoding calculado sobre a base inteira.

    Por padrao, substituimos a coluna original codificada pela versao TE_*. Isso
    impede que o modelo interprete um codigo numerico como escala ordinal.
    """

    def __init__(self, columns: list[str] | None = None, smoothing: float = 20.0):
        self.columns = columns or []
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_df = pd.DataFrame(X).copy()
        y_series = pd.Series(y, index=X_df.index).astype(float)
        self.global_mean_ = float(y_series.mean())
        self.mapping_ = {}

        for col in self.columns:
            if col not in X_df.columns:
                continue
            key = X_df[col].astype("object").where(X_df[col].notna(), "__MISSING__")
            stats = y_series.groupby(key).agg(["mean", "count"])
            smooth = (
                (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing)
                / (stats["count"] + self.smoothing)
            )
            self.mapping_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col in self.columns:
            if col not in X_df.columns or col not in self.mapping_:
                continue
            key = X_df[col].astype("object").where(X_df[col].notna(), "__MISSING__")
            X_df[f"TE_{col}"] = key.map(self.mapping_[col]).fillna(self.global_mean_).astype(float)
            X_df = X_df.drop(columns=[col])
        return X_df


def numeric_model_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Mantem apenas colunas numericas para modelos que aceitam NaN nativo."""

    return X.select_dtypes(include="number").copy()


def optional_dependency_status(
    output_dir: Path,
    rows: list[dict[str, str]],
    table_name: str,
) -> None:
    """Salva e imprime status de dependencias opcionais."""

    status = pd.DataFrame(rows)
    save_table(status, output_dir, table_name)
    if not status.empty:
        print(f"\n=== {table_name} ===")
        print(status.to_string(index=False))


def make_xgboost_classifier(config: Config, fast: bool, **overrides):
    """Cria XGBClassifier com defaults consistentes para os experimentos."""

    from xgboost import XGBClassifier

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": config.random_state,
        "n_estimators": 180 if fast else 360,
        "n_jobs": -1,
        "verbosity": 0,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def make_lightgbm_classifier(config: Config, fast: bool, **overrides):
    """Cria LGBMClassifier com defaults consistentes para os experimentos."""

    from lightgbm import LGBMClassifier

    params = {
        "objective": "binary",
        "random_state": config.random_state,
        "n_estimators": 200 if fast else 380,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


def evaluate_with_validated_thresholds(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    experiment_name: str,
) -> list[dict[str, float | str]]:
    """Avalia modelo em 0.50 e thresholds escolhidos em validacao interna."""

    thresholds, threshold_report = select_thresholds_on_validation_multi(
        name,
        estimator,
        X_train,
        y_train,
        output_dir,
        config,
    )
    save_table(
        threshold_report,
        output_dir,
        f"{experiment_name}_threshold_search_{name.lower()}",
    )

    rows = []
    for threshold_name, threshold in {"0.50": 0.50, **thresholds}.items():
        metrics, _ = evaluate_model(
            f"{name}@{threshold_name}",
            estimator,
            X_test,
            y_test,
            threshold=threshold,
        )
        metrics["experimento"] = experiment_name
        rows.append(metrics)
    return rows


def stratified_sample_frame(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Gera amostra estratificada para validacoes rapidas."""

    if len(X) <= max_rows:
        return X.copy(), y.copy()
    sample_idx, _ = train_test_split(
        X.index,
        train_size=max_rows,
        stratify=y,
        random_state=random_state,
    )
    return X.loc[sample_idx].copy(), y.loc[sample_idx].copy()


def validate_disagreements(
    df_model: pd.DataFrame,
    cleanup_metadata: dict[str, object],
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Valida empiricamente pontos de discordancia discutidos.

    Comparacoes:
    1. StandardScaler vs RobustScaler no KNN.
    2. Manter vs remover ANOSULTIMADECLARACAO + flag.
    3. Custo potencial de criar uma flag para cada coluna CASA.

    Esta etapa usa amostra estratificada para nao transformar a validacao de
    decisoes em um treinamento completo caro.
    """

    X = df_model.drop(columns=[config.target_col])
    y = df_model[config.target_col]
    max_rows = 12_000 if fast else 25_000
    X_cmp, y_cmp = stratified_sample_frame(X, y, max_rows, config.random_state)
    # Usamos os mesmos folds do protocolo principal. Isso deixa a comparacao das
    # discordancias alinhada com o GridSearchCV final. Em `--fast`, reduzimos a
    # amostra, mas nao mudamos o desenho experimental.
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    rows = []

    for scaler_name, scaler in [("standard", StandardScaler()), ("robust", RobustScaler())]:
        numeric_cols = X_cmp.select_dtypes(include="number").columns.tolist()
        categorical_cols = X_cmp.select_dtypes(include=["object", "string"]).columns.tolist()
        preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler)
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=21,
                        weights="distance",
                        metric="manhattan",
                    ),
                ),
            ]
        )
        scores = cross_validate(
            model,
            X_cmp,
            y_cmp,
            cv=cv,
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        rows.append(
            {
                "discordancia": "scaler_knn",
                "configuracao": scaler_name,
                "average_precision_mean": scores["test_average_precision"].mean(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )

    for keep_decl in [True, False]:
        X_variant = X_cmp.copy()
        if not keep_decl:
            X_variant = X_variant.drop(
                columns=["ANOSULTIMADECLARACAO", "DECLARACAO_AUSENTE"],
                errors="ignore",
            )
        numeric_cols = X_variant.select_dtypes(include="number").columns.tolist()
        categorical_cols = X_variant.select_dtypes(include=["object", "string"]).columns.tolist()
        preprocessor = build_preprocessor(numeric_cols, categorical_cols, StandardScaler())
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=10,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
        scores = cross_validate(
            model,
            X_variant,
            y_cmp,
            cv=cv,
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        rows.append(
            {
                "discordancia": "anos_ultima_declaracao",
                "configuracao": "manter_com_flag" if keep_decl else "remover_coluna_e_flag",
                "average_precision_mean": scores["test_average_precision"].mean(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )

    existing_casa_cols = cleanup_metadata["existing_casa_cols"]
    rows.append(
        {
            "discordancia": "indicadores_casa",
            "configuracao": f"flag_manual=1; indicadores_CASA_potenciais={len(existing_casa_cols)}",
            "average_precision_mean": np.nan,
            "roc_auc_mean": np.nan,
        }
    )

    report = pd.DataFrame(rows)
    save_table(report, output_dir, "decision_validation_report")
    print("\n=== Validacao de discordancias ===")
    print(report.to_string(index=False))
    return report


def split_train_test(
    df_model: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Cria holdout estratificado.

    Esta e a barreira de leakage. Depois dela, `X_test` so aparece na avaliacao
    final. Imputacao e normalizacao acontecem dentro do pipeline.
    """

    X = df_model.drop(columns=[config.target_col])
    y = df_model[config.target_col].astype(int)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )


def build_model_pipelines(
    X_train: pd.DataFrame,
    config: Config,
) -> tuple[Pipeline, Pipeline, DummyClassifier]:
    """Cria baseline, pipeline KNN e pipeline Arvore de Decisao.

    A primeira rodada de validacao empirica mostrou que `StandardScaler` foi
    melhor que `RobustScaler` para o KNN neste dataset. Por isso, o pipeline
    final usa `StandardScaler` como normalizador padrao.

    Criamos preprocessadores separados para KNN e Arvore. O `GridSearchCV`
    clonaria internamente os objetos, entao compartilhar o mesmo objeto nao
    causaria vazamento. Ainda assim, objetos separados deixam a inspecao do
    pipeline mais clara e evitam confusao depois do fit.
    """

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    print(f"Features numericas: {len(numeric_cols)}")
    print(f"Features categoricas: {len(categorical_cols)}")
    if not categorical_cols:
        print(
            "Apos excluir variaveis sensiveis, nao restaram categoricas textuais permitidas."
        )

    knn_preprocessor = build_preprocessor(numeric_cols, categorical_cols, StandardScaler())
    dt_preprocessor = build_preprocessor(numeric_cols, categorical_cols, StandardScaler())

    dummy = DummyClassifier(strategy="most_frequent")
    pipe_knn = Pipeline(
        [
            ("preprocess", knn_preprocessor),
            ("clf", KNeighborsClassifier()),
        ]
    )
    pipe_dt = Pipeline(
        [
            ("preprocess", dt_preprocessor),
            (
                "clf",
                DecisionTreeClassifier(
                    random_state=config.random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return pipe_knn, pipe_dt, dummy


def build_enhanced_model_pipelines(
    X_train: pd.DataFrame,
    config: Config,
) -> tuple[Pipeline, Pipeline, DummyClassifier]:
    """Cria pipelines da trilha enhanced.

    Diferencas em relacao a trilha original:

    - KNN recebe `VarianceThreshold` e `SelectKBest(f_classif)` para reduzir
      dimensionalidade antes de calcular distancias.
    - Arvore permite que `class_weight` seja buscado no grid, comparando
      `"balanced"` com pesos manuais menos ou mais agressivos.
    """

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    knn_preprocessor = build_preprocessor(numeric_cols, categorical_cols, StandardScaler())
    dt_preprocessor = build_preprocessor(numeric_cols, categorical_cols, StandardScaler())

    dummy = DummyClassifier(strategy="most_frequent")
    pipe_knn = Pipeline(
        [
            ("preprocess", knn_preprocessor),
            ("variance", VarianceThreshold()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("clf", KNeighborsClassifier()),
        ]
    )
    pipe_dt = Pipeline(
        [
            ("preprocess", dt_preprocessor),
            ("variance", VarianceThreshold()),
            (
                "clf",
                DecisionTreeClassifier(random_state=config.random_state),
            ),
        ]
    )
    return pipe_knn, pipe_dt, dummy


def run_grid_searches(
    pipe_knn: Pipeline,
    pipe_dt: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> tuple[GridSearchCV, GridSearchCV]:
    """Executa GridSearchCV para KNN e Arvore.

    O grid do KNN e propositalmente conservador porque KNN e caro na predicao:
    ele calcula distancias contra muitos exemplos. A Arvore tem grid um pouco
    mais amplo, mas `class_weight="balanced"` fica fixo no estimador.
    """

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    if fast:
        param_grid_knn = {
            # Mesmo em modo rapido mantemos mais de uma combinacao. Assim,
            # continua sendo uma busca real, nao apenas cross-validation de um
            # hiperparametro fixo.
            "clf__n_neighbors": [11, 21],
            "clf__metric": ["manhattan"],
            "clf__weights": ["distance"],
        }
        param_grid_dt = {
            "clf__max_depth": [5, 10],
            "clf__min_samples_split": [10, 50],
            "clf__criterion": ["gini"],
        }
    else:
        param_grid_knn = {
            "clf__n_neighbors": [11, 21],
            "clf__metric": ["euclidean", "manhattan"],
            "clf__weights": ["uniform", "distance"],
        }
        param_grid_dt = {
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_split": [2, 10, 50],
            "clf__criterion": ["gini", "entropy"],
        }

    gs_knn = GridSearchCV(
        pipe_knn,
        param_grid_knn,
        cv=cv,
        scoring=config.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs_dt = GridSearchCV(
        pipe_dt,
        param_grid_dt,
        cv=cv,
        scoring=config.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    gs_knn.fit(X_train, y_train)
    gs_dt.fit(X_train, y_train)

    knn_results = grid_results_table(gs_knn, "KNN")
    dt_results = grid_results_table(gs_dt, "DecisionTree")
    save_table(knn_results, output_dir, "grid_results_knn")
    save_table(dt_results, output_dir, "grid_results_decision_tree")

    print("\n=== GridSearchCV ===")
    print(f"Melhor KNN | AP medio CV: {gs_knn.best_score_:.4f} | params: {gs_knn.best_params_}")
    print(f"Melhor DT  | AP medio CV: {gs_dt.best_score_:.4f} | params: {gs_dt.best_params_}")
    return gs_knn, gs_dt


def run_enhanced_grid_searches(
    pipe_knn: Pipeline,
    pipe_dt: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> tuple[GridSearchCV, GridSearchCV]:
    """Executa GridSearchCV da trilha enhanced.

    Esta busca nao substitui a original. Ela testa melhorias adicionais:

    - KNN com selecao de features para reduzir ruido em distancia.
    - Arvore com `min_samples_leaf`, `max_leaf_nodes`, `ccp_alpha` e pesos de
      classe alternativos.

    O grid completo foi mantido deliberadamente moderado. Arvores sao baratas,
    mas grids enormes podem tornar o Colab instavel.
    """

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    if fast:
        param_grid_knn = {
            "selector__k": [30, "all"],
            "clf__n_neighbors": [21],
            "clf__metric": ["manhattan"],
            "clf__weights": ["distance"],
        }
        param_grid_dt = [
            {
                "clf__criterion": ["entropy"],
                "clf__max_depth": [8, 10],
                "clf__min_samples_split": [50, 100],
                "clf__min_samples_leaf": [10, 25],
                "clf__max_leaf_nodes": [None, 200],
                "clf__ccp_alpha": [0.0],
                "clf__class_weight": ["balanced", {0: 1, 1: 5}],
            }
        ]
    else:
        param_grid_knn = {
            "selector__k": [20, 35, 50, "all"],
            "clf__n_neighbors": [11, 21, 41],
            "clf__metric": ["manhattan"],
            "clf__weights": ["distance"],
        }
        param_grid_dt = [
            {
                "clf__criterion": ["entropy"],
                "clf__max_depth": [8, 10, 12],
                "clf__min_samples_split": [50, 100, 200],
                "clf__min_samples_leaf": [10, 25, 50],
                "clf__max_leaf_nodes": [None],
                "clf__ccp_alpha": [0.0, 0.0001],
                "clf__class_weight": ["balanced"],
            },
            {
                "clf__criterion": ["entropy"],
                "clf__max_depth": [10, 12],
                "clf__min_samples_split": [50, 100],
                "clf__min_samples_leaf": [10, 25],
                "clf__max_leaf_nodes": [None, 200],
                "clf__ccp_alpha": [0.0],
                "clf__class_weight": [
                    {0: 1, 1: 3},
                    {0: 1, 1: 5},
                    {0: 1, 1: 7},
                ],
            },
        ]

    gs_knn = GridSearchCV(
        pipe_knn,
        param_grid_knn,
        cv=cv,
        scoring=config.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs_dt = GridSearchCV(
        pipe_dt,
        param_grid_dt,
        cv=cv,
        scoring=config.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    gs_knn.fit(X_train, y_train)
    gs_dt.fit(X_train, y_train)

    knn_results = grid_results_table(gs_knn, "EnhancedKNN")
    dt_results = grid_results_table(gs_dt, "EnhancedDecisionTree")
    save_table(knn_results, output_dir, "enhanced_grid_results_knn")
    save_table(dt_results, output_dir, "enhanced_grid_results_decision_tree")

    print("\n=== GridSearchCV enhanced ===")
    print(f"Melhor enhanced KNN | AP medio CV: {gs_knn.best_score_:.4f} | params: {gs_knn.best_params_}")
    print(f"Melhor enhanced DT  | AP medio CV: {gs_dt.best_score_:.4f} | params: {gs_dt.best_params_}")
    return gs_knn, gs_dt


def run_advanced_grid_searches(
    pipe_knn: Pipeline,
    pipe_dt: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> tuple[GridSearchCV, GridSearchCV]:
    """Executa busca refinada da trilha advanced.

    A busca da Arvore fica concentrada ao redor do melhor resultado enhanced:
    `entropy`, `max_depth` perto de 10, `min_samples_leaf` em torno de 25 e
    pesos de classe entre 4x e 6x para a classe 1.
    """

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    if fast:
        param_grid_knn = {
            "selector__k": [35],
            "clf__n_neighbors": [41],
            "clf__metric": ["manhattan"],
            "clf__weights": ["distance"],
        }
        param_grid_dt = {
            "clf__criterion": ["entropy"],
            "clf__max_depth": [10, 11],
            "clf__min_samples_split": [75, 100],
            "clf__min_samples_leaf": [25, 35],
            "clf__max_leaf_nodes": [200],
            "clf__ccp_alpha": [0.0],
            "clf__class_weight": [{0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}],
        }
    else:
        param_grid_knn = {
            "selector__k": [25, 35, 45],
            "clf__n_neighbors": [31, 41, 61],
            "clf__metric": ["manhattan"],
            "clf__weights": ["distance"],
        }
        param_grid_dt = {
            "clf__criterion": ["entropy"],
            "clf__max_depth": [9, 10, 11],
            "clf__min_samples_split": [75, 100, 150],
            "clf__min_samples_leaf": [15, 25, 35],
            "clf__max_leaf_nodes": [150, 200, 300],
            "clf__ccp_alpha": [0.0, 0.00005],
            "clf__class_weight": [{0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}],
        }

    gs_knn = GridSearchCV(
        pipe_knn,
        param_grid_knn,
        cv=cv,
        scoring=config.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs_dt = GridSearchCV(
        pipe_dt,
        param_grid_dt,
        cv=cv,
        scoring=config.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    gs_knn.fit(X_train, y_train)
    gs_dt.fit(X_train, y_train)

    save_table(grid_results_table(gs_knn, "AdvancedKNN"), output_dir, "advanced_grid_results_knn")
    save_table(
        grid_results_table(gs_dt, "AdvancedDecisionTree"),
        output_dir,
        "advanced_grid_results_decision_tree",
    )

    print("\n=== GridSearchCV advanced ===")
    print(f"Melhor advanced KNN | AP medio CV: {gs_knn.best_score_:.4f} | params: {gs_knn.best_params_}")
    print(f"Melhor advanced DT  | AP medio CV: {gs_dt.best_score_:.4f} | params: {gs_dt.best_params_}")
    return gs_knn, gs_dt


def grid_results_table(grid: GridSearchCV, model_name: str) -> pd.DataFrame:
    """Organiza resultados do GridSearchCV em tabela curta."""

    result = pd.DataFrame(grid.cv_results_).copy()
    param_cols = [col for col in result.columns if col.startswith("param_")]
    cols = ["rank_test_score", "mean_test_score", "std_test_score", *param_cols]
    result = result[cols].sort_values("rank_test_score").reset_index(drop=True)
    result.insert(0, "modelo", model_name)
    return result


def get_positive_proba(model, X_eval: pd.DataFrame) -> np.ndarray:
    """Retorna probabilidade da classe positiva com protecao para baseline."""

    proba = model.predict_proba(X_eval)
    if proba.shape[1] == 1:
        only_class = model.classes_[0]
        return np.ones(len(X_eval)) if only_class == 1 else np.zeros(len(X_eval))
    positive_index = list(model.classes_).index(1)
    return proba[:, positive_index]


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict[str, float | str], np.ndarray]:
    """Avalia um modelo no holdout."""

    y_proba = get_positive_proba(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "modelo": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_1": precision_score(y_test, y_pred, zero_division=0),
        "recall_1": recall_score(y_test, y_pred, zero_division=0),
        "f1_1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    for key, value in metrics.items():
        if key == "modelo":
            continue
        print(f"{key}: {value:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusao:")
    print(matrix)
    return metrics, matrix


def threshold_analysis(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    """Analisa precision, recall e F1 em diferentes thresholds.

    Esta tabela e diagnostica. O threshold operacional final nao deve ser
    escolhido repetidamente olhando o teste; em um projeto de producao, ele
    seria escolhido em validacao ou por custo de negocio. Aqui, a analise ajuda
    a explicar por que a Arvore detecta muitos inadimplentes, mas tambem gera
    muitos falsos positivos em threshold 0.5.
    """

    y_proba = get_positive_proba(model, X_test)
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "modelo": name,
                "threshold": round(float(threshold), 2),
                "precision_1": precision_score(y_test, y_pred, zero_division=0),
                "recall_1": recall_score(y_test, y_pred, zero_division=0),
                "f1_1": f1_score(y_test, y_pred, zero_division=0),
                "predicted_positive_rate": float(y_pred.mean()),
            }
        )

    report = pd.DataFrame(rows)
    safe_name = name.lower().replace(" ", "_")
    save_table(report, output_dir, f"threshold_analysis_{safe_name}")
    return report


def select_threshold_on_validation(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    config: Config,
) -> tuple[float, pd.DataFrame]:
    """Escolhe threshold usando uma validacao interna do treino.

    Esta funcao evita escolher threshold olhando o holdout. Ela separa uma
    validacao interna a partir de `X_train`, refita uma copia do melhor pipeline
    nessa subamostra de treino, mede precision/recall/F1 na validacao interna e
    escolhe o threshold com maior F1 da classe 1.
    """

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=config.random_state,
    )

    base_estimator = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
    threshold_model = clone(base_estimator)
    threshold_model.fit(X_fit, y_fit)
    y_proba = get_positive_proba(threshold_model, X_val)

    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "modelo": name,
                "threshold": round(float(threshold), 2),
                "precision_1": precision_score(y_val, y_pred, zero_division=0),
                "recall_1": recall_score(y_val, y_pred, zero_division=0),
                "f1_1": f1_score(y_val, y_pred, zero_division=0),
                "predicted_positive_rate": float(y_pred.mean()),
            }
        )

    report = pd.DataFrame(rows)
    best_row = report.sort_values(
        ["f1_1", "recall_1", "precision_1"],
        ascending=False,
    ).iloc[0]
    safe_name = name.lower().replace(" ", "_")
    save_table(report, output_dir, f"validated_threshold_search_{safe_name}")
    return float(best_row["threshold"]), report


def select_thresholds_on_validation_multi(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    config: Config,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Escolhe thresholds por F1, F2 e custo em validacao interna.

    F2 da mais peso ao recall do que ao precision. Isso e coerente com credito
    quando perder inadimplentes e mais caro do que revisar falsos positivos.

    A regra de custo minimiza:

        custo = fn_cost * falsos_negativos + fp_cost * falsos_positivos

    O holdout nao participa dessa escolha.
    """

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=config.random_state,
    )

    base_estimator = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
    threshold_model = clone(base_estimator)
    threshold_model.fit(X_fit, y_fit)
    y_proba = get_positive_proba(threshold_model, X_val)

    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        rows.append(
            {
                "modelo": name,
                "threshold": round(float(threshold), 2),
                "precision_1": precision_score(y_val, y_pred, zero_division=0),
                "recall_1": recall_score(y_val, y_pred, zero_division=0),
                "f1_1": f1_score(y_val, y_pred, zero_division=0),
                "f2_1": fbeta_score(y_val, y_pred, beta=2, zero_division=0),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
                "business_cost": float(fn_cost * fn + fp_cost * fp),
                "predicted_positive_rate": float(y_pred.mean()),
            }
        )

    report = pd.DataFrame(rows)
    thresholds = {
        "f1": float(report.sort_values(["f1_1", "recall_1"], ascending=False).iloc[0]["threshold"]),
        "f2": float(report.sort_values(["f2_1", "recall_1"], ascending=False).iloc[0]["threshold"]),
        "cost": float(report.sort_values(["business_cost", "recall_1"], ascending=[True, False]).iloc[0]["threshold"]),
    }
    safe_name = name.lower().replace(" ", "_")
    save_table(report, output_dir, f"advanced_validated_threshold_search_{safe_name}")
    return thresholds, report


def make_calibrated_classifier(estimator, method: str, cv: int):
    """Cria CalibratedClassifierCV compativel com versoes do sklearn."""

    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def run_modeling(
    df_model: pd.DataFrame,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Executa split, baseline, GridSearchCV e avaliacao final."""

    X_train, X_test, y_train, y_test = split_train_test(df_model, config)
    print(f"X_train: {X_train.shape} | TARGET=1: {y_train.mean():.4f}")
    print(f"X_test:  {X_test.shape} | TARGET=1: {y_test.mean():.4f}")

    pipe_knn, pipe_dt, dummy = build_model_pipelines(X_train, config)
    dummy.fit(X_train, y_train)
    gs_knn, gs_dt = run_grid_searches(
        pipe_knn,
        pipe_dt,
        X_train,
        y_train,
        output_dir,
        config,
        fast,
    )

    final_rows = []
    matrices = {}
    for name, model in [
        ("DummyClassifier", dummy),
        ("KNN", gs_knn),
        ("DecisionTree", gs_dt),
    ]:
        metrics, matrix = evaluate_model(name, model, X_test, y_test)
        metrics["experimento"] = "original"
        final_rows.append(metrics)
        matrices[name] = matrix

    threshold_reports = []
    for name, model in [("KNN", gs_knn), ("DecisionTree", gs_dt)]:
        report = threshold_analysis(name, model, X_test, y_test, output_dir)
        threshold_reports.append(report)
        best_f1_row = report.sort_values("f1_1", ascending=False).iloc[0]
        print(
            f"\nMelhor threshold diagnostico por F1 para {name}: "
            f"t={best_f1_row['threshold']:.2f}, "
            f"precision={best_f1_row['precision_1']:.4f}, "
            f"recall={best_f1_row['recall_1']:.4f}, "
            f"f1={best_f1_row['f1_1']:.4f}"
        )

    if threshold_reports:
        save_table(
            pd.concat(threshold_reports, ignore_index=True),
            output_dir,
            "threshold_analysis_all_models",
        )

    final_metrics = pd.DataFrame(final_rows).sort_values(
        "average_precision",
        ascending=False,
    )
    save_table(final_metrics, output_dir, "final_holdout_metrics")

    for name, matrix in matrices.items():
        pd.DataFrame(matrix).to_csv(
            output_dir / "tables" / f"confusion_matrix_{name.lower()}.csv",
            index=False,
        )

    print("\n=== Ranking final no holdout ===")
    print(final_metrics.to_string(index=False))
    return final_metrics


def run_enhanced_modeling(
    df_model: pd.DataFrame,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Executa a trilha enhanced mantendo comparabilidade com a original."""

    df_enhanced, feature_metadata = add_enhanced_features(df_model, output_dir, config)
    print("\n=== Feature engineering enhanced ===")
    print(feature_metadata.to_string(index=False))

    X_train, X_test, y_train, y_test = split_train_test(df_enhanced, config)
    print(f"Enhanced X_train: {X_train.shape} | TARGET=1: {y_train.mean():.4f}")
    print(f"Enhanced X_test:  {X_test.shape} | TARGET=1: {y_test.mean():.4f}")

    pipe_knn, pipe_dt, dummy = build_enhanced_model_pipelines(X_train, config)
    dummy.fit(X_train, y_train)
    gs_knn, gs_dt = run_enhanced_grid_searches(
        pipe_knn,
        pipe_dt,
        X_train,
        y_train,
        output_dir,
        config,
        fast,
    )

    selected_thresholds = {}
    threshold_search_reports = []
    for name, model in [("EnhancedKNN", gs_knn), ("EnhancedDecisionTree", gs_dt)]:
        threshold, report = select_threshold_on_validation(
            name,
            model,
            X_train,
            y_train,
            output_dir,
            config,
        )
        selected_thresholds[name] = threshold
        threshold_search_reports.append(report)
        print(f"Threshold validado para {name}: {threshold:.2f}")

    if threshold_search_reports:
        save_table(
            pd.concat(threshold_search_reports, ignore_index=True),
            output_dir,
            "enhanced_validated_threshold_search_all",
        )

    final_rows = []
    matrices = {}
    evaluation_plan = [
        ("EnhancedDummyClassifier", dummy, 0.50),
        ("EnhancedKNN@0.50", gs_knn, 0.50),
        ("EnhancedKNN@validated_threshold", gs_knn, selected_thresholds["EnhancedKNN"]),
        ("EnhancedDecisionTree@0.50", gs_dt, 0.50),
        (
            "EnhancedDecisionTree@validated_threshold",
            gs_dt,
            selected_thresholds["EnhancedDecisionTree"],
        ),
    ]

    for name, model, threshold in evaluation_plan:
        metrics, matrix = evaluate_model(name, model, X_test, y_test, threshold=threshold)
        metrics["experimento"] = "enhanced"
        final_rows.append(metrics)
        matrices[name] = matrix

    threshold_reports = []
    for name, model in [("EnhancedKNN", gs_knn), ("EnhancedDecisionTree", gs_dt)]:
        report = threshold_analysis(name, model, X_test, y_test, output_dir)
        threshold_reports.append(report)

    if threshold_reports:
        save_table(
            pd.concat(threshold_reports, ignore_index=True),
            output_dir,
            "enhanced_threshold_analysis_all_models",
        )

    final_metrics = pd.DataFrame(final_rows).sort_values(
        "average_precision",
        ascending=False,
    )
    save_table(final_metrics, output_dir, "enhanced_final_holdout_metrics")

    for name, matrix in matrices.items():
        safe_name = name.lower().replace("@", "_").replace(" ", "_")
        pd.DataFrame(matrix).to_csv(
            output_dir / "tables" / f"enhanced_confusion_matrix_{safe_name}.csv",
            index=False,
        )

    print("\n=== Ranking enhanced no holdout ===")
    print(final_metrics.to_string(index=False))
    return final_metrics


def run_ensemble_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Executa benchmarks complementares fora da rubrica principal.

    Estes modelos nao substituem a comparacao obrigatoria KNN vs Arvore de
    Decisao. Eles funcionam como teto pratico adicional: se ensembles e boosting
    melhorarem bastante, isso mostra que o dataset tem interacoes nao lineares
    que uma unica arvore nao consegue explorar com estabilidade.

    LightGBM e XGBoost entram apenas quando estiverem instalados. Assim, o
    script continua rodando no Colab basico com sklearn puro, mas permite um
    experimento mais proximo do mercado quando as dependencias opcionais
    estiverem disponiveis.
    """

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, StandardScaler())

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    hist_kwargs = {
        "random_state": config.random_state,
        "max_iter": 120 if fast else 220,
        "early_stopping": True,
        "validation_fraction": 0.10,
    }
    hist_param_grid = {
        "clf__learning_rate": [0.05, 0.10],
        "clf__max_leaf_nodes": [31, 63] if fast else [31, 63, 127],
        "clf__l2_regularization": [0.0, 0.1] if fast else [0.0, 0.05, 0.1],
    }
    if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
        hist_param_grid["clf__class_weight"] = [{0: 1, 1: 5}, "balanced"]

    models = [
        (
            "AdvancedRandomForest",
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=250 if fast else 400,
                            random_state=config.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__max_depth": [10, 14] if fast else [10, 14, None],
                "clf__min_samples_leaf": [25, 50],
                "clf__class_weight": [{0: 1, 1: 5}, "balanced"],
            },
        ),
        (
            "AdvancedExtraTrees",
            Pipeline(
                [
                    ("preprocess", clone(preprocessor)),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=250 if fast else 400,
                            random_state=config.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__max_depth": [10, 14] if fast else [10, 14, None],
                "clf__min_samples_leaf": [25, 50],
                "clf__class_weight": [{0: 1, 1: 5}, "balanced"],
            },
        ),
        (
            "AdvancedHistGradientBoosting",
            Pipeline(
                [
                    ("preprocess", clone(preprocessor)),
                    ("clf", HistGradientBoostingClassifier(**hist_kwargs)),
                ]
            ),
            hist_param_grid,
        ),
    ]

    optional_rows = []
    try:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "AdvancedLightGBM",
                Pipeline(
                    [
                        ("preprocess", clone(preprocessor)),
                        (
                            "clf",
                            LGBMClassifier(
                                objective="binary",
                                random_state=config.random_state,
                                n_estimators=180 if fast else 350,
                                n_jobs=-1,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
                {
                    "clf__learning_rate": [0.05, 0.10],
                    "clf__num_leaves": [31, 63] if fast else [31, 63, 127],
                    "clf__max_depth": [-1, 8] if fast else [-1, 6, 10],
                    "clf__scale_pos_weight": [5, 8, 10] if fast else [4, 6, 8, 10],
                },
            )
        )
        optional_rows.append(
            {"modelo": "AdvancedLightGBM", "status": "incluido", "motivo": "lightgbm instalado"}
        )
    except Exception as exc:
        optional_rows.append(
            {
                "modelo": "AdvancedLightGBM",
                "status": "pulado",
                "motivo": f"indisponivel ({exc}); instale com: ../bin/python -m pip install lightgbm",
            }
        )

    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "AdvancedXGBoost",
                Pipeline(
                    [
                        ("preprocess", clone(preprocessor)),
                        (
                            "clf",
                            XGBClassifier(
                                objective="binary:logistic",
                                eval_metric="logloss",
                                tree_method="hist",
                                random_state=config.random_state,
                                n_estimators=160 if fast else 320,
                                n_jobs=-1,
                                verbosity=0,
                            ),
                        ),
                    ]
                ),
                {
                    "clf__learning_rate": [0.05, 0.10],
                    "clf__max_depth": [3, 5] if fast else [3, 5, 7],
                    "clf__subsample": [0.8, 1.0],
                    "clf__colsample_bytree": [0.8, 1.0],
                    "clf__scale_pos_weight": [5, 8, 10] if fast else [4, 6, 8, 10],
                },
            )
        )
        optional_rows.append(
            {"modelo": "AdvancedXGBoost", "status": "incluido", "motivo": "xgboost instalado"}
        )
    except Exception as exc:
        optional_rows.append(
            {
                "modelo": "AdvancedXGBoost",
                "status": "pulado",
                "motivo": f"indisponivel ({exc}); instale com: ../bin/python -m pip install xgboost",
            }
        )

    optional_status = pd.DataFrame(optional_rows)
    save_table(optional_status, output_dir, "optional_boosting_dependency_status")
    if not optional_status.empty:
        print("\n=== Dependencias opcionais de boosting ===")
        print(optional_status.to_string(index=False))

    rows = []
    for name, pipeline, param_grid in models:
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=config.scoring,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        try:
            grid.fit(X_train, y_train)
        except Exception as exc:
            rows.append(
                {
                    "modelo": name,
                    "threshold": np.nan,
                    "accuracy": np.nan,
                    "precision_1": np.nan,
                    "recall_1": np.nan,
                    "f1_1": np.nan,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "experimento": "benchmark_ensemble",
                    "erro": str(exc)[:500],
                }
            )
            print(f"\n{name} pulado por erro durante o fit: {exc}")
            continue
        save_table(grid_results_table(grid, name), output_dir, f"benchmark_grid_results_{name.lower()}")

        thresholds, threshold_report_df = select_thresholds_on_validation_multi(
            name,
            grid,
            X_train,
            y_train,
            output_dir,
            config,
        )
        save_table(threshold_report_df, output_dir, f"benchmark_threshold_search_{name.lower()}")

        for threshold_name, threshold in {"0.50": 0.50, **thresholds}.items():
            metrics, _ = evaluate_model(
                f"{name}@{threshold_name}",
                grid,
                X_test,
                y_test,
                threshold=threshold,
            )
            metrics["experimento"] = "benchmark_ensemble"
            rows.append(metrics)

    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "benchmark_ensemble_holdout_metrics")
    return report


def run_resampling_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Testa SMOTE e NearMiss como experimentos opcionais.

    Reamostragem muda a distribuicao observada pelo classificador, entao ela
    precisa ficar dentro de um `imblearn.pipeline.Pipeline`. Desse modo, cada
    fold do GridSearchCV aplica SMOTE/NearMiss somente no treino daquele fold,
    nunca na validacao e nunca no holdout.

    Este bloco responde a pergunta metodologica: o foco esta apenas em feature
    engineering/thresholds ou reamostragem tambem ajuda? Por isso mantemos KNN e
    Arvore como classificadores, mas alteramos a distribuicao do treino.
    """

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import NearMiss
    except Exception as exc:
        status = pd.DataFrame(
            [
                {
                    "experimento": "resampling",
                    "status": "pulado",
                    "motivo": (
                        f"imbalanced-learn indisponivel ({exc}); instale com: "
                        "../bin/python -m pip install imbalanced-learn"
                    ),
                }
            ]
        )
        save_table(status, output_dir, "resampling_dependency_status")
        print("\n=== Reamostragem pulada ===")
        print(status.to_string(index=False))
        return pd.DataFrame()

    status = pd.DataFrame(
        [
            {
                "experimento": "resampling",
                "status": "incluido",
                "motivo": "imbalanced-learn instalado; SMOTE e NearMiss serao avaliados dentro do pipeline",
            }
        ]
    )
    save_table(status, output_dir, "resampling_dependency_status")

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    sampler_specs = [
        (
            "SMOTE",
            SMOTE(
                random_state=config.random_state,
                sampling_strategy=0.50,
                k_neighbors=5,
            ),
            {
                "sampler__sampling_strategy": [0.35, 0.50] if not fast else [0.50],
            },
        ),
        (
            "NearMiss",
            NearMiss(
                version=1,
                sampling_strategy=0.50,
                n_neighbors=3,
            ),
            {
                "sampler__sampling_strategy": [0.35, 0.50] if not fast else [0.50],
            },
        ),
    ]

    model_specs = []
    for sampler_name, sampler, sampler_grid in sampler_specs:
        knn_pipe = ImbPipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(numeric_cols, categorical_cols, StandardScaler()),
                ),
                ("variance", VarianceThreshold()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("sampler", sampler),
                ("clf", KNeighborsClassifier()),
            ]
        )
        knn_grid = {
            **sampler_grid,
            "selector__k": [35] if fast else [35, 45],
            "clf__n_neighbors": [41] if fast else [31, 41, 61],
            "clf__metric": ["manhattan"],
            "clf__weights": ["distance"],
        }
        model_specs.append((f"ResamplingKNN_{sampler_name}", knn_pipe, knn_grid))

        dt_pipe = ImbPipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(numeric_cols, categorical_cols, StandardScaler()),
                ),
                ("variance", VarianceThreshold()),
                ("sampler", clone(sampler)),
                (
                    "clf",
                    DecisionTreeClassifier(random_state=config.random_state),
                ),
            ]
        )
        dt_grid = {
            **sampler_grid,
            "clf__criterion": ["entropy"],
            "clf__max_depth": [10, 11] if fast else [9, 10, 11],
            "clf__min_samples_split": [100] if fast else [100, 150],
            "clf__min_samples_leaf": [25, 35],
            "clf__max_leaf_nodes": [200],
            "clf__ccp_alpha": [0.0, 0.00005] if not fast else [0.0],
            # Com reamostragem, comparamos sem class_weight e com pesos suaves.
            # Isso evita somar dois mecanismos agressivos contra a classe 0.
            "clf__class_weight": [None, {0: 1, 1: 3}],
        }
        model_specs.append((f"ResamplingDecisionTree_{sampler_name}", dt_pipe, dt_grid))

    rows = []
    threshold_reports = []
    for name, pipeline, param_grid in model_specs:
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=config.scoring,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        try:
            grid.fit(X_train, y_train)
        except Exception as exc:
            rows.append(
                {
                    "modelo": name,
                    "threshold": np.nan,
                    "accuracy": np.nan,
                    "precision_1": np.nan,
                    "recall_1": np.nan,
                    "f1_1": np.nan,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "experimento": "resampling",
                    "erro": str(exc)[:500],
                }
            )
            print(f"\n{name} pulado por erro durante o fit: {exc}")
            continue

        safe_name = name.lower()
        save_table(grid_results_table(grid, name), output_dir, f"resampling_grid_results_{safe_name}")
        print(f"\nMelhor {name} | AP medio CV: {grid.best_score_:.4f} | params: {grid.best_params_}")

        thresholds, threshold_report = select_thresholds_on_validation_multi(
            name,
            grid,
            X_train,
            y_train,
            output_dir,
            config,
        )
        threshold_reports.append(threshold_report)

        for threshold_name, threshold in {"0.50": 0.50, **thresholds}.items():
            metrics, _ = evaluate_model(
                f"{name}@{threshold_name}",
                grid,
                X_test,
                y_test,
                threshold=threshold,
            )
            metrics["experimento"] = "resampling"
            rows.append(metrics)

    if threshold_reports:
        save_table(
            pd.concat(threshold_reports, ignore_index=True),
            output_dir,
            "resampling_threshold_search_all",
        )

    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "resampling_holdout_metrics")
    print("\n=== Ranking resampling no holdout ===")
    print(report.to_string(index=False))
    return report


def run_native_boosting_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Testa boosting preservando NaN, sem imputacao e sem scaler.

    XGBoost e LightGBM conseguem aprender rotas especificas para valores
    ausentes. Como a ausencia e informativa nesta base, esta trilha compara o
    pipeline sklearn imputado contra modelos que veem os NaNs diretamente.
    """

    X_train_num = numeric_model_frame(X_train)
    X_test_num = numeric_model_frame(X_test)
    dropped_cols = sorted(set(X_train.columns) - set(X_train_num.columns))
    save_table(
        pd.DataFrame(
            [
                ("features_entrada", X_train.shape[1]),
                ("features_numericas_usadas", X_train_num.shape[1]),
                ("colunas_nao_numericas_descartadas", dropped_cols),
                ("estrategia", "NaN preservado; sem SimpleImputer; sem scaler"),
            ],
            columns=["chave", "valor"],
        ),
        output_dir,
        "native_boosting_metadata",
    )

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    models = []
    status_rows = []
    try:
        models.append(
            (
                "NativeXGBoost",
                make_xgboost_classifier(config, fast),
                {
                    "learning_rate": [0.03, 0.05] if not fast else [0.05],
                    "max_depth": [3, 5],
                    "min_child_weight": [1, 5] if not fast else [5],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8, 1.0] if not fast else [0.8],
                    # O grid anterior mostrou que 4 venceu 8/10. Refinamos ao
                    # redor dessa regiao em vez de assumir a razao bruta 9.46.
                    "scale_pos_weight": [2, 3, 4, 5, 6] if not fast else [4],
                    "reg_lambda": [1.0, 3.0] if not fast else [1.0],
                },
            )
        )
        status_rows.append({"modelo": "NativeXGBoost", "status": "incluido", "motivo": "xgboost instalado"})
    except Exception as exc:
        status_rows.append({"modelo": "NativeXGBoost", "status": "pulado", "motivo": str(exc)})

    try:
        models.append(
            (
                "NativeLightGBM",
                make_lightgbm_classifier(config, fast),
                {
                    "learning_rate": [0.03, 0.05] if not fast else [0.05],
                    "num_leaves": [31, 63] if not fast else [63],
                    "max_depth": [-1, 6, 10] if not fast else [-1],
                    "min_child_samples": [20, 50] if not fast else [50],
                    "subsample": [0.8, 1.0] if not fast else [0.8],
                    "colsample_bytree": [0.8, 1.0] if not fast else [0.8],
                    "scale_pos_weight": [2, 3, 4, 5, 6] if not fast else [4],
                    "reg_lambda": [0.0, 1.0] if not fast else [1.0],
                },
            )
        )
        status_rows.append({"modelo": "NativeLightGBM", "status": "incluido", "motivo": "lightgbm instalado"})
    except Exception as exc:
        status_rows.append({"modelo": "NativeLightGBM", "status": "pulado", "motivo": str(exc)})

    optional_dependency_status(output_dir, status_rows, "native_boosting_dependency_status")

    rows = []
    for name, estimator, param_grid in models:
        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=config.scoring,
            n_jobs=1,
            refit=True,
            verbose=1,
        )
        try:
            grid.fit(X_train_num, y_train)
        except Exception as exc:
            rows.append(
                {
                    "modelo": name,
                    "threshold": np.nan,
                    "accuracy": np.nan,
                    "precision_1": np.nan,
                    "recall_1": np.nan,
                    "f1_1": np.nan,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "experimento": "native_boosting",
                    "erro": str(exc)[:500],
                }
            )
            print(f"\n{name} pulado por erro durante o fit: {exc}")
            continue

        save_table(grid_results_table(grid, name), output_dir, f"native_boosting_grid_results_{name.lower()}")
        print(f"\nMelhor {name} | AP medio CV: {grid.best_score_:.4f} | params: {grid.best_params_}")
        rows.extend(
            evaluate_with_validated_thresholds(
                name,
                grid,
                X_train_num,
                X_test_num,
                y_train,
                y_test,
                output_dir,
                config,
                "native_boosting",
            )
        )

    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "native_boosting_holdout_metrics")
    print("\n=== Ranking native boosting no holdout ===")
    print(report.to_string(index=False))
    return report


def run_optuna_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Otimiza XGBoost nativo com Optuna.

    O objetivo e maximizar Average Precision em validacao cruzada. O holdout
    continua fora do processo de busca e so entra na avaliacao final.
    """

    try:
        import optuna
        from xgboost import XGBClassifier
    except Exception as exc:
        status = pd.DataFrame(
            [
                {
                    "experimento": "optuna_xgboost",
                    "status": "pulado",
                    "motivo": f"dependencia indisponivel ({exc}); instale optuna e xgboost",
                }
            ]
        )
        save_table(status, output_dir, "optuna_xgboost_status")
        print("\n=== Optuna XGBoost pulado ===")
        print(status.to_string(index=False))
        return pd.DataFrame()

    X_train_num = numeric_model_frame(X_train)
    X_test_num = numeric_model_frame(X_test)
    cv = StratifiedKFold(
        n_splits=3 if fast else config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    n_trials = 8 if fast else 45

    def objective(trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": config.random_state,
            "n_jobs": -1,
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 180, 520),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 7.0),
        }
        model = XGBClassifier(**params)
        scores = cross_validate(
            model,
            X_train_num,
            y_train,
            cv=cv,
            scoring=config.scoring,
            n_jobs=1,
        )
        return float(scores["test_score"].mean())

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = study.trials_dataframe()
    save_table(trials, output_dir, "optuna_xgboost_trials")

    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": config.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
    )
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train_num, y_train)

    save_table(
        pd.DataFrame(
            [{"best_cv_average_precision": study.best_value, **study.best_params}]
        ),
        output_dir,
        "optuna_xgboost_best_params",
    )
    print(f"\nMelhor Optuna XGBoost | AP medio CV: {study.best_value:.4f} | params: {study.best_params}")

    rows = evaluate_with_validated_thresholds(
        "OptunaXGBoost",
        best_model,
        X_train_num,
        X_test_num,
        y_train,
        y_test,
        output_dir,
        config,
        "optuna_xgboost",
    )
    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "optuna_xgboost_holdout_metrics")
    print("\n=== Ranking Optuna XGBoost no holdout ===")
    print(report.to_string(index=False))
    return report


def build_xgboost_importance_ranking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame | None:
    """Gera ranking de features via SHAP quando possivel, ou importance nativa."""

    try:
        model = make_xgboost_classifier(
            config,
            fast,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4,
            reg_lambda=1.0,
        )
    except Exception as exc:
        save_table(
            pd.DataFrame([{"status": "pulado", "motivo": str(exc)}]),
            output_dir,
            "importance_selection_status",
        )
        return None

    X_train_num = numeric_model_frame(X_train)
    model.fit(X_train_num, y_train)

    method = "feature_importances_"
    try:
        import shap

        sample_size = min(5_000 if not fast else 1_500, len(X_train_num))
        sample = X_train_num.sample(sample_size, random_state=config.random_state)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        importance = np.abs(shap_values).mean(axis=0)
        method = "shap_mean_abs"
    except Exception:
        importance = model.feature_importances_

    ranking = (
        pd.DataFrame(
            {
                "feature": X_train_num.columns,
                "importance": importance,
                "method": method,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    save_table(ranking, output_dir, "importance_selection_feature_ranking")
    return ranking


def run_importance_selection_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Testa XGBoost com subconjuntos top-N por importancia."""

    ranking = build_xgboost_importance_ranking(X_train, y_train, output_dir, config, fast)
    if ranking is None or ranking.empty:
        return pd.DataFrame()

    X_train_num = numeric_model_frame(X_train)
    X_test_num = numeric_model_frame(X_test)
    top_values = [30, 50, 70, "all"] if not fast else [35, "all"]
    cv = StratifiedKFold(
        n_splits=3 if fast else config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    rows = []
    cv_rows = []
    for top_n in top_values:
        selected_features = (
            ranking["feature"].tolist()
            if top_n == "all"
            else ranking.head(int(top_n))["feature"].tolist()
        )
        model = make_xgboost_classifier(
            config,
            fast,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4,
            reg_lambda=1.0,
        )
        scores = cross_validate(
            model,
            X_train_num[selected_features],
            y_train,
            cv=cv,
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        cv_rows.append(
            {
                "top_n": top_n,
                "qtd_features": len(selected_features),
                "cv_average_precision": scores["test_average_precision"].mean(),
                "cv_roc_auc": scores["test_roc_auc"].mean(),
            }
        )
        model.fit(X_train_num[selected_features], y_train)
        rows.extend(
            evaluate_with_validated_thresholds(
                f"ImportanceXGBoost_top{top_n}",
                model,
                X_train_num[selected_features],
                X_test_num[selected_features],
                y_train,
                y_test,
                output_dir,
                config,
                "importance_selection",
            )
        )

    save_table(pd.DataFrame(cv_rows), output_dir, "importance_selection_cv_summary")
    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "importance_selection_holdout_metrics")
    print("\n=== Ranking importance selection no holdout ===")
    print(report.to_string(index=False))
    return report


def run_stacking_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Combina modelos fortes via StackingClassifier."""

    X_train_num = numeric_model_frame(X_train)
    X_test_num = numeric_model_frame(X_test)
    estimators = []
    status_rows = []

    try:
        estimators.append(
            (
                "xgb",
                make_xgboost_classifier(
                    config,
                    fast,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=4,
                ),
            )
        )
        status_rows.append({"modelo": "xgb", "status": "incluido", "motivo": "xgboost instalado"})
    except Exception as exc:
        status_rows.append({"modelo": "xgb", "status": "pulado", "motivo": str(exc)})

    try:
        estimators.append(
            (
                "lgbm",
                make_lightgbm_classifier(
                    config,
                    fast,
                    learning_rate=0.05,
                    num_leaves=63,
                    scale_pos_weight=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
            )
        )
        status_rows.append({"modelo": "lgbm", "status": "incluido", "motivo": "lightgbm instalado"})
    except Exception as exc:
        status_rows.append({"modelo": "lgbm", "status": "pulado", "motivo": str(exc)})

    estimators.append(
        (
            "hgb",
            HistGradientBoostingClassifier(
                random_state=config.random_state,
                max_iter=120 if fast else 220,
                learning_rate=0.05,
                max_leaf_nodes=63,
                early_stopping=True,
            ),
        )
    )
    status_rows.append({"modelo": "hgb", "status": "incluido", "motivo": "sklearn nativo"})
    optional_dependency_status(output_dir, status_rows, "stacking_dependency_status")

    if len(estimators) < 2:
        return pd.DataFrame()

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=config.random_state,
        ),
        stack_method="predict_proba",
        cv=3 if fast else config.cv_folds,
        n_jobs=-1,
        passthrough=False,
    )
    stack.fit(X_train_num, y_train)

    rows = evaluate_with_validated_thresholds(
        "StackingBoosting",
        stack,
        X_train_num,
        X_test_num,
        y_train,
        y_test,
        output_dir,
        config,
        "stacking",
    )
    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "stacking_holdout_metrics")
    print("\n=== Ranking stacking no holdout ===")
    print(report.to_string(index=False))
    return report


def detect_target_encoding_candidates(X_train: pd.DataFrame) -> pd.DataFrame:
    """Identifica possiveis codigos categoricos numericos.

    A heuristica e conservadora: a coluna precisa ter nome de codigo/localidade
    e baixa cardinalidade relativa. Colunas como MEDIARENDACEP ou IDHMUNICIPIO
    nao entram automaticamente apenas por conterem CEP/MUNICIPIO no nome.
    """

    include_tokens = ("COD", "CODIGO", "CEP", "MUNICIPIO", "CIDADE", "BAIRRO", "UF", "DDD")
    exclude_prefixes = (
        "MEDIA",
        "PERCENT",
        "IDH",
        "PIB",
        "IDADE",
        "QTD",
        "TOTAL",
        "RAZAO",
        "LOG",
        "DIST",
        "SOMA",
        "MAIOR",
        "MENOR",
    )
    exclude_tokens = (
        "MEDIA",
        "MEDIO",
        "RENDA",
        "IDADE",
        "PERCENT",
        "IDH",
        "PIB",
        "DIST",
    )
    rows = []
    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            continue
        upper_col = col.upper()
        if upper_col.startswith(exclude_prefixes):
            continue
        if any(token in upper_col for token in exclude_tokens):
            continue
        if not any(token in upper_col for token in include_tokens):
            continue
        series = X_train[col].dropna()
        if series.empty:
            continue
        unique_count = int(series.nunique())
        unique_ratio = float(unique_count / len(series))
        integer_like = bool(np.allclose(series, np.round(series)))
        is_candidate = integer_like and 10 <= unique_count <= 5_000 and unique_ratio <= 0.20
        rows.append(
            {
                "coluna": col,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "integer_like": integer_like,
                "candidate": is_candidate,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["coluna", "unique_count", "unique_ratio", "integer_like", "candidate"]
        )
    return pd.DataFrame(rows).sort_values(["candidate", "unique_count"], ascending=[False, False])


def run_target_encoding_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    config: Config,
    fast: bool,
) -> pd.DataFrame:
    """Testa target encoding apenas quando houver codigos categoricos provaveis."""

    candidates = detect_target_encoding_candidates(X_train)
    save_table(candidates, output_dir, "target_encoding_candidate_report")
    selected = candidates.query("candidate == True")["coluna"].tolist() if not candidates.empty else []

    if not selected:
        status = pd.DataFrame(
            [
                {
                    "experimento": "target_encoding",
                    "status": "pulado",
                    "motivo": "Nenhuma coluna numerica parece codigo categorico seguro para target encoding.",
                }
            ]
        )
        save_table(status, output_dir, "target_encoding_status")
        print("\n=== Target encoding pulado ===")
        print(status.to_string(index=False))
        return pd.DataFrame()

    try:
        xgb = make_xgboost_classifier(config, fast)
    except Exception as exc:
        status = pd.DataFrame(
            [{"experimento": "target_encoding", "status": "pulado", "motivo": str(exc)}]
        )
        save_table(status, output_dir, "target_encoding_status")
        return pd.DataFrame()

    pipeline = Pipeline(
        [
            ("target_encoder", TargetMeanEncoder(columns=selected, smoothing=20.0)),
            ("clf", xgb),
        ]
    )
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    param_grid = {
        "target_encoder__smoothing": [10.0, 30.0] if not fast else [20.0],
        "clf__learning_rate": [0.05],
        "clf__max_depth": [3, 5] if not fast else [5],
        "clf__min_child_weight": [1, 5] if not fast else [5],
        "clf__scale_pos_weight": [3, 4, 5] if not fast else [4],
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=config.scoring,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    save_table(grid_results_table(grid, "TargetEncodedXGBoost"), output_dir, "target_encoding_grid_results")
    save_table(
        pd.DataFrame({"target_encoded_columns": selected}),
        output_dir,
        "target_encoding_selected_columns",
    )

    rows = evaluate_with_validated_thresholds(
        "TargetEncodedXGBoost",
        grid,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir,
        config,
        "target_encoding",
    )
    report = pd.DataFrame(rows).sort_values(["average_precision", "f1_1"], ascending=False)
    save_table(report, output_dir, "target_encoding_holdout_metrics")
    print("\n=== Ranking target encoding no holdout ===")
    print(report.to_string(index=False))
    return report


def run_advanced_modeling(
    df_model: pd.DataFrame,
    output_dir: Path,
    config: Config,
    fast: bool,
    benchmark_ensembles: bool,
    resampling_models: bool,
    native_boosting: bool,
    optuna_xgboost: bool,
    importance_selection: bool,
    stacking_models: bool,
    target_encoding: bool,
) -> pd.DataFrame:
    """Executa a terceira trilha experimental."""

    df_advanced, feature_metadata = add_advanced_features(df_model, output_dir, config)
    print("\n=== Feature engineering advanced ===")
    print(feature_metadata.to_string(index=False))

    X_train, X_test, y_train, y_test = split_train_test(df_advanced, config)
    print(f"Advanced X_train: {X_train.shape} | TARGET=1: {y_train.mean():.4f}")
    print(f"Advanced X_test:  {X_test.shape} | TARGET=1: {y_test.mean():.4f}")

    pipe_knn, pipe_dt, dummy = build_enhanced_model_pipelines(X_train, config)
    dummy.fit(X_train, y_train)
    gs_knn, gs_dt = run_advanced_grid_searches(
        pipe_knn,
        pipe_dt,
        X_train,
        y_train,
        output_dir,
        config,
        fast,
    )

    threshold_reports = []
    selected_thresholds = {}
    for name, model in [("AdvancedKNN", gs_knn), ("AdvancedDecisionTree", gs_dt)]:
        thresholds, report = select_thresholds_on_validation_multi(
            name,
            model,
            X_train,
            y_train,
            output_dir,
            config,
        )
        selected_thresholds[name] = thresholds
        threshold_reports.append(report)
        print(f"Thresholds validados para {name}: {thresholds}")

    if threshold_reports:
        save_table(
            pd.concat(threshold_reports, ignore_index=True),
            output_dir,
            "advanced_validated_threshold_search_all",
        )

    final_rows = []
    matrices = {}
    evaluation_plan = [
        ("AdvancedDummyClassifier", dummy, 0.50),
        ("AdvancedKNN@0.50", gs_knn, 0.50),
        ("AdvancedKNN@threshold_f1", gs_knn, selected_thresholds["AdvancedKNN"]["f1"]),
        ("AdvancedKNN@threshold_f2", gs_knn, selected_thresholds["AdvancedKNN"]["f2"]),
        ("AdvancedKNN@threshold_cost", gs_knn, selected_thresholds["AdvancedKNN"]["cost"]),
        ("AdvancedDecisionTree@0.50", gs_dt, 0.50),
        (
            "AdvancedDecisionTree@threshold_f1",
            gs_dt,
            selected_thresholds["AdvancedDecisionTree"]["f1"],
        ),
        (
            "AdvancedDecisionTree@threshold_f2",
            gs_dt,
            selected_thresholds["AdvancedDecisionTree"]["f2"],
        ),
        (
            "AdvancedDecisionTree@threshold_cost",
            gs_dt,
            selected_thresholds["AdvancedDecisionTree"]["cost"],
        ),
    ]

    calibrated_models = []
    for method in (["sigmoid"] if fast else ["sigmoid", "isotonic"]):
        calibrated = make_calibrated_classifier(
            clone(gs_dt.best_estimator_),
            method=method,
            cv=3,
        )
        calibrated.fit(X_train, y_train)
        model_name = f"AdvancedDecisionTreeCalibrated_{method}"
        thresholds, report = select_thresholds_on_validation_multi(
            model_name,
            calibrated,
            X_train,
            y_train,
            output_dir,
            config,
        )
        save_table(report, output_dir, f"advanced_calibrated_threshold_search_{method}")
        calibrated_models.append((model_name, calibrated, thresholds))

    for model_name, calibrated, thresholds in calibrated_models:
        evaluation_plan.extend(
            [
                (f"{model_name}@0.50", calibrated, 0.50),
                (f"{model_name}@threshold_f1", calibrated, thresholds["f1"]),
                (f"{model_name}@threshold_f2", calibrated, thresholds["f2"]),
                (f"{model_name}@threshold_cost", calibrated, thresholds["cost"]),
            ]
        )

    for name, model, threshold in evaluation_plan:
        metrics, matrix = evaluate_model(name, model, X_test, y_test, threshold=threshold)
        metrics["experimento"] = "advanced"
        final_rows.append(metrics)
        matrices[name] = matrix

    for name, model in [("AdvancedKNN", gs_knn), ("AdvancedDecisionTree", gs_dt), *[(n, m) for n, m, _ in calibrated_models]]:
        report = threshold_analysis(name, model, X_test, y_test, output_dir)
        save_table(report, output_dir, f"advanced_threshold_analysis_{name.lower()}")

    final_metrics = pd.DataFrame(final_rows).sort_values(
        ["average_precision", "f1_1"],
        ascending=False,
    )
    save_table(final_metrics, output_dir, "advanced_final_holdout_metrics")

    for name, matrix in matrices.items():
        safe_name = (
            name.lower()
            .replace("@", "_")
            .replace(" ", "_")
            .replace(":", "")
        )
        pd.DataFrame(matrix).to_csv(
            output_dir / "tables" / f"advanced_confusion_matrix_{safe_name}.csv",
            index=False,
        )

    if benchmark_ensembles:
        benchmark_report = run_ensemble_benchmarks(
            X_train,
            X_test,
            y_train,
            y_test,
            output_dir,
            config,
            fast,
        )
        final_metrics = pd.concat([final_metrics, benchmark_report], ignore_index=True, sort=False)
        final_metrics = final_metrics.sort_values(["average_precision", "f1_1"], ascending=False)
        save_table(final_metrics, output_dir, "advanced_final_plus_benchmarks_holdout_metrics")

    if resampling_models:
        resampling_report = run_resampling_benchmarks(
            X_train,
            X_test,
            y_train,
            y_test,
            output_dir,
            config,
            fast,
        )
        if not resampling_report.empty:
            final_metrics = pd.concat([final_metrics, resampling_report], ignore_index=True, sort=False)
            final_metrics = final_metrics.sort_values(["average_precision", "f1_1"], ascending=False)
            save_table(final_metrics, output_dir, "advanced_final_plus_resampling_holdout_metrics")

    optional_reports = []
    if native_boosting:
        optional_reports.append(
            run_native_boosting_benchmarks(
                X_train,
                X_test,
                y_train,
                y_test,
                output_dir,
                config,
                fast,
            )
        )

    if optuna_xgboost:
        optional_reports.append(
            run_optuna_xgboost(
                X_train,
                X_test,
                y_train,
                y_test,
                output_dir,
                config,
                fast,
            )
        )

    if importance_selection:
        optional_reports.append(
            run_importance_selection_benchmarks(
                X_train,
                X_test,
                y_train,
                y_test,
                output_dir,
                config,
                fast,
            )
        )

    if stacking_models:
        optional_reports.append(
            run_stacking_benchmarks(
                X_train,
                X_test,
                y_train,
                y_test,
                output_dir,
                config,
                fast,
            )
        )

    if target_encoding:
        optional_reports.append(
            run_target_encoding_benchmarks(
                X_train,
                X_test,
                y_train,
                y_test,
                output_dir,
                config,
                fast,
            )
        )

    optional_reports = [report for report in optional_reports if report is not None and not report.empty]
    if optional_reports:
        final_metrics = pd.concat([final_metrics, *optional_reports], ignore_index=True, sort=False)
        final_metrics = final_metrics.sort_values(["average_precision", "f1_1"], ascending=False)
        save_table(final_metrics, output_dir, "advanced_final_plus_experimental_holdout_metrics")

    if benchmark_ensembles and resampling_models:
        save_table(final_metrics, output_dir, "advanced_final_plus_all_optional_holdout_metrics")

    print("\n=== Ranking advanced no holdout ===")
    print(final_metrics.to_string(index=False))
    return final_metrics


def run_pipeline(args: argparse.Namespace) -> None:
    """Orquestra as etapas selecionadas por CLI."""

    output_dir = ensure_output_dir(args.output_dir)
    data_path = resolve_path(args.data_path)
    df_raw = load_raw_data(data_path)

    should_run_eda = args.all or args.eda
    should_validate = args.all or args.validate_decisions
    should_model = args.all or args.model
    should_enhanced_model = args.all or args.enhanced_model
    should_native_boosting = args.advanced_extras or args.native_boosting
    should_optuna_xgboost = args.advanced_extras or args.optuna_xgboost
    should_importance_selection = args.advanced_extras or args.importance_selection
    should_stacking_models = args.advanced_extras or args.stacking_models
    should_target_encoding = args.advanced_extras or args.target_encoding
    should_advanced_model = (
        args.all
        or args.advanced_model
        or args.benchmark_ensembles
        or args.resampling_models
        or should_native_boosting
        or should_optuna_xgboost
        or should_importance_selection
        or should_stacking_models
        or should_target_encoding
    )

    if not any([
        should_run_eda,
        should_validate,
        should_model,
        should_enhanced_model,
        should_advanced_model,
    ]):
        print(
            "Nenhuma etapa selecionada. Use --eda, --validate-decisions, "
            "--model, --enhanced-model, --advanced-model, --benchmark-ensembles, "
            "--resampling-models, --advanced-extras ou --all."
        )
        return

    if should_run_eda:
        present_dataset(df_raw, output_dir, CONFIG)
        df_eda = make_eda_frame(df_raw, CONFIG)
        run_univariate_eda(df_eda, output_dir, CONFIG)
        run_multivariate_eda(df_eda, output_dir, CONFIG)

    if should_validate or should_model or should_enhanced_model or should_advanced_model:
        df_model, cleanup_metadata = deterministic_cleanup(df_raw, CONFIG)
        metadata_df = pd.DataFrame(
            [(key, str(value)) for key, value in cleanup_metadata.items()],
            columns=["chave", "valor"],
        )
        save_table(metadata_df, output_dir, "cleanup_metadata")

        print("\n=== Cleanup deterministico ===")
        print(metadata_df.to_string(index=False))
        remaining_nan = (
            df_model.isna()
            .sum()
            .loc[lambda s: s > 0]
            .sort_values(ascending=False)
            .rename("qtd_nan_restante")
            .reset_index()
            .rename(columns={"index": "coluna"})
        )
        save_table(remaining_nan, output_dir, "remaining_nan_after_cleanup")

        if should_validate:
            validate_disagreements(
                df_model,
                cleanup_metadata,
                output_dir,
                CONFIG,
                args.fast,
            )

        original_metrics = None
        enhanced_metrics = None
        advanced_metrics = None

        if should_model:
            original_metrics = run_modeling(df_model, output_dir, CONFIG, args.fast)

        if should_enhanced_model:
            enhanced_metrics = run_enhanced_modeling(df_model, output_dir, CONFIG, args.fast)

        if should_advanced_model:
            advanced_metrics = run_advanced_modeling(
                df_model,
                output_dir,
                CONFIG,
                args.fast,
                args.benchmark_ensembles,
                args.resampling_models,
                should_native_boosting,
                should_optuna_xgboost,
                should_importance_selection,
                should_stacking_models,
                should_target_encoding,
            )

        comparison_frames = [
            frame for frame in [original_metrics, enhanced_metrics, advanced_metrics]
            if frame is not None
        ]
        if len(comparison_frames) >= 2:
            comparison = pd.concat(
                comparison_frames,
                ignore_index=True,
                sort=False,
            ).sort_values(["average_precision", "f1_1"], ascending=False)
            save_table(comparison, output_dir, "comparison_all_experiments")
            print("\n=== Comparativo de experimentos ===")
            print(comparison.to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")


if __name__ == "__main__":
    run_pipeline(parse_args())
