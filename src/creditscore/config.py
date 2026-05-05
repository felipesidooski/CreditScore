"""Project constants used by preprocessing, training and inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Stores reproducible experiment settings.

    Attributes:
        random_state: Seed used in train/test splits and models.
        target_col: Name of the binary target column.
        test_size: Holdout test size used in the reported experiments.
        sentinels: Encoded missing values found in the original dataset.
    """

    random_state: int = 42
    target_col: str = "TARGET"
    test_size: float = 0.20
    sentinels: tuple[int, ...] = (-99, -9998, -9999)


CONFIG = ExperimentConfig()

LEGAL_SENSITIVE_COLS = ["ORIENTACAO_SEXUAL", "RELIGIAO"]

HIGH_MISSING_COLS = [
    "ANOSULTIMARESTITUICAO",
    "ANOSULTIMADECLARACAOPAGAR",
]

ID_COLS = ["HS_CPF"]

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

FINAL_MODEL_METRICS = {
    "importance_xgboost_top50": {
        "threshold": 0.35,
        "accuracy": 0.7229,
        "precision_1": 0.1800,
        "recall_1": 0.5335,
        "f1_1": 0.2692,
        "roc_auc": 0.6967,
        "average_precision": 0.2020,
    },
    "optuna_xgboost": {
        "threshold": 0.25,
        "accuracy": 0.7947,
        "precision_1": 0.2094,
        "recall_1": 0.4132,
        "f1_1": 0.2780,
        "roc_auc": 0.6990,
        "average_precision": 0.2006,
    },
}

