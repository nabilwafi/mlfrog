"""Artifact paths. Repo root = package root; blobs under artifacts/."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = ROOT / "configs"
DATA = ROOT / "artifacts"

RAW = DATA / "raw"
FEATURES = DATA / "features"
LABELS = DATA / "labels"
DATASETS = DATA / "datasets"
MODELS = DATA / "models"
DIAGNOSTICS = DATA / "diagnostics"
RESEARCH = DATA / "research"
CONTEXT = DATA / "context"
REPORTS = DATA / "reports"
PAPER = DATA / "paper_trading"
LOGS = DATA / "logs"

MODELS_V3_LONG = DATA / "models_v3" / "long"
LIGHTGBM_LONG_V3 = MODELS_V3_LONG / "lightgbm_long_v3.pkl"
LIGHTGBM_LONG_V3_TEST_PREDS = MODELS_V3_LONG / "lightgbm_long_v3_test_preds.parquet"
LIGHTGBM_LONG_V3_FULLTEST_PREDS = MODELS_V3_LONG / "lightgbm_long_v3_fulltest_preds.parquet"

CALIBRATOR_LONG_V3 = DATA / "models" / "calibration" / "confidence_calibrator_long_v3.pkl"

FEATURES_V3 = FEATURES / "xauusd_h1_h4_d1_features_v3.parquet"
TRIPLE_BARRIER_LABELS = LABELS / "xauusd_triple_barrier_labels.parquet"

H1_RAW = RAW / "XAUUSD_H1.csv"
H4_RAW = RAW / "XAUUSD_H4.csv"
D1_RAW = RAW / "XAUUSD_D1.csv"


def raw_symbol_dir(symbol: str, timeframe: str) -> Path:
    """Storage: artifacts/raw/{SYMBOL}/{TF}/."""
    return RAW / symbol.upper() / timeframe.upper()


def model_symbol_dir(symbol: str, timeframe: str, side: str) -> Path:
    """Storage: artifacts/models/{SYMBOL}/{TF}/{side}/."""
    return MODELS / symbol.upper() / timeframe.upper() / side.lower()


def diagnostics_symbol_dir(symbol: str, timeframe: str, side: str) -> Path:
    """Storage: artifacts/diagnostics/{SYMBOL}/{TF}/{side}/."""
    return DIAGNOSTICS / symbol.upper() / timeframe.upper() / side.lower()


def research_symbol_dir(symbol: str, timeframe: str, side: str) -> Path:
    """Storage: artifacts/research/{SYMBOL}/{TF}/{side}/."""
    return RESEARCH / symbol.upper() / timeframe.upper() / side.lower()


MT5_CONFIG = CONFIGS / "config.yaml"
MT5_CONFIG_EXAMPLE = CONFIGS / "config.example.yaml"
