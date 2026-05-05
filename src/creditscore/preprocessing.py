"""Preprocessing and feature engineering classes for the Credit Score project."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditscore.config import (
    CASA_COLS,
    CONFIG,
    HIGH_MISSING_COLS,
    ID_COLS,
    LEGAL_SENSITIVE_COLS,
    LOG_FEATURE_COLS,
    ZERO_FLAG_COLS,
    ExperimentConfig,
)


class CreditScorePreprocessor(BaseEstimator, TransformerMixin):
    """Applies deterministic cleanup to the raw credit score dataset.

    Args:
        config: Experiment settings such as target name and sentinel values.

    Attributes:
        removed_columns_: Columns dropped during the latest transform.
        existing_casa_cols_: Household columns found in the input data.
    """

    def __init__(self, config: ExperimentConfig = CONFIG):
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Returns the deterministic transformer instance.

        Args:
            X: Raw input dataframe.
            y: Optional target series, ignored.

        Returns:
            The fitted transformer instance.
        """

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cleans raw data and creates missingness flags.

        Args:
            X: Raw dataframe with or without the TARGET column.

        Returns:
            Dataframe with sentinels converted to NaN, legal/sensitive columns
            removed, extreme-missing columns removed and deterministic flags
            added.
        """

        df = X.copy()
        df = df.replace(list(self.config.sentinels), np.nan)

        existing_casa_cols = [col for col in CASA_COLS if col in df.columns]
        self.existing_casa_cols_ = existing_casa_cols
        if existing_casa_cols:
            df["GRUPO_CASA_AUSENTE"] = df[existing_casa_cols].isna().any(axis=1).astype(int)
        else:
            df["GRUPO_CASA_AUSENTE"] = 0

        if "ANOSULTIMADECLARACAO" in df.columns:
            df["DECLARACAO_AUSENTE"] = df["ANOSULTIMADECLARACAO"].isna().astype(int)
        else:
            df["DECLARACAO_AUSENTE"] = 0

        cols_to_drop = [
            col
            for col in [*ID_COLS, *LEGAL_SENSITIVE_COLS, *HIGH_MISSING_COLS]
            if col in df.columns
        ]
        self.removed_columns_ = cols_to_drop
        return df.drop(columns=cols_to_drop)


class CreditScoreFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates the enhanced and advanced deterministic features.

    Args:
        config: Experiment settings such as target name.

    Attributes:
        created_features_: Feature names created during the latest transform.
    """

    def __init__(self, config: ExperimentConfig = CONFIG):
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Returns the feature engineering transformer instance.

        Args:
            X: Input dataframe after deterministic cleanup.
            y: Optional target series, ignored.

        Returns:
            The fitted transformer instance.
        """

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adds advanced features used by the champion models.

        Args:
            X: Cleaned dataframe with or without TARGET.

        Returns:
            Dataframe with missingness aggregates, log transforms, zero flags
            and ratio features.
        """

        df = X.copy()
        original_cols = set(df.columns)
        feature_cols = [col for col in df.columns if col != self.config.target_col]

        df["QTD_CAMPOS_AUSENTES"] = df[feature_cols].isna().sum(axis=1)
        df["PCT_CAMPOS_AUSENTES"] = df[feature_cols].isna().mean(axis=1)

        self._add_group_missing_features(df, CASA_COLS, "CASA")
        self._add_suffix_missing_features(df, "CEP")
        self._add_suffix_missing_features(df, "MUNICIPIO")
        self._add_log_features(df)
        self._add_zero_flags(df)
        self._add_ratio_features(df)
        self._add_vehicle_features(df)

        self.created_features_ = sorted(set(df.columns) - original_cols)
        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fits and transforms the input dataframe.

        Args:
            X: Input dataframe after deterministic cleanup.
            y: Optional target series, ignored.

        Returns:
            Dataframe with engineered features.
        """

        return self.fit(X, y).transform(X)

    def _add_group_missing_features(self, df: pd.DataFrame, columns: Iterable[str], prefix: str) -> None:
        existing = [col for col in columns if col in df.columns]
        if not existing:
            return
        missing = df[existing].isna()
        df[f"{prefix}_QTD_AUSENTES"] = missing.sum(axis=1)
        df[f"{prefix}_TOTALMENTE_AUSENTE"] = missing.all(axis=1).astype(int)
        df[f"{prefix}_PARCIALMENTE_AUSENTE"] = (
            missing.any(axis=1) & ~missing.all(axis=1)
        ).astype(int)

    def _add_suffix_missing_features(self, df: pd.DataFrame, suffix: str) -> None:
        cols = [col for col in df.columns if col.endswith(suffix) and col != self.config.target_col]
        if not cols:
            return
        missing = df[cols].isna()
        df[f"{suffix}_QTD_AUSENTES"] = missing.sum(axis=1)
        df[f"{suffix}_AUSENTE"] = missing.any(axis=1).astype(int)

    def _add_log_features(self, df: pd.DataFrame) -> None:
        for col in LOG_FEATURE_COLS:
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"LOG_{col}"] = np.log1p(numeric.clip(lower=0))

    def _add_zero_flags(self, df: pd.DataFrame) -> None:
        for col in ZERO_FLAG_COLS:
            if col not in df.columns:
                continue
            df[f"{col}_ZERO"] = (pd.to_numeric(df[col], errors="coerce") == 0).astype(int)

    def _add_ratio_features(self, df: pd.DataFrame) -> None:
        self._add_ratio(df, "RAZAO_RENDA_ESTIMADA_CEP", "ESTIMATIVARENDA", "MEDIARENDACEP")
        self._add_ratio(df, "RAZAO_RENDA_ESTIMADA_CASA", "ESTIMATIVARENDA", "MEDIARENDACASA")
        self._add_ratio(df, "RAZAO_RENDA_CASA_CEP", "MEDIARENDACASA", "MEDIARENDACEP")
        self._add_ratio(df, "RENDA_TOTAL_CASA_POR_PESSOA", "SOMARENDACASA", "QTDPESSOASCASA")
        self._add_ratio(df, "RAZAO_DIST_RISCO_CENTRO", "DISTZONARISCO", "DISTCENTROCIDADE")
        self._add_ratio(df, "RAZAO_RENDA_CEP_PIB", "MEDIARENDACEP", "PIBMUNICIPIO")
        self._add_ratio(df, "RAZAO_RENDA_ESTIMADA_IDH", "ESTIMATIVARENDA", "IDHMUNICIPIO")

    def _add_vehicle_features(self, df: pd.DataFrame) -> None:
        vehicle_cols = [
            "QTDUTILITARIOMUNICIPIO",
            "QTDAUTOMOVELMUNICIPIO",
            "QTDCAMINHAOMUNICIPIO",
            "QTDCAMINHONETEMUNICIPIO",
            "QTDMOTOMUNICIPIO",
        ]
        existing = [col for col in vehicle_cols if col in df.columns]
        if not existing:
            return
        df["TOTAL_VEICULOS_MUNICIPIO"] = (
            df[existing].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        )
        self._add_ratio(df, "RAZAO_RENDA_ESTIMADA_VEICULOS", "ESTIMATIVARENDA", "TOTAL_VEICULOS_MUNICIPIO")

    @staticmethod
    def _add_ratio(df: pd.DataFrame, new_col: str, numerator_col: str, denominator_col: str) -> None:
        if numerator_col not in df.columns or denominator_col not in df.columns:
            return
        numerator = pd.to_numeric(df[numerator_col], errors="coerce")
        denominator = pd.to_numeric(df[denominator_col], errors="coerce").replace(0, np.nan)
        ratio = numerator / denominator
        df[new_col] = ratio.replace([np.inf, -np.inf], np.nan)


class CreditScoreDataPipeline:
    """Runs the full deterministic preparation used by final models.

    Args:
        config: Experiment settings.

    Attributes:
        preprocessor: Deterministic cleanup component.
        feature_engineer: Feature engineering component.
    """

    def __init__(self, config: ExperimentConfig = CONFIG):
        self.config = config
        self.preprocessor = CreditScorePreprocessor(config)
        self.feature_engineer = CreditScoreFeatureEngineer(config)

    def prepare(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Prepares raw rows for model training or inference.

        Args:
            df_raw: Raw dataframe, optionally including TARGET.

        Returns:
            Prepared dataframe with deterministic cleanup and advanced features.
        """

        cleaned = self.preprocessor.fit_transform(df_raw)
        return self.feature_engineer.fit_transform(cleaned)

    def split_X_y(self, df_prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Splits prepared labeled data into features and target.

        Args:
            df_prepared: Prepared dataframe containing TARGET.

        Returns:
            Tuple with feature dataframe and integer target series.
        """

        if self.config.target_col not in df_prepared.columns:
            raise ValueError(f"Column {self.config.target_col!r} not found.")
        X = df_prepared.drop(columns=[self.config.target_col])
        y = df_prepared[self.config.target_col].astype(int)
        return X, y
