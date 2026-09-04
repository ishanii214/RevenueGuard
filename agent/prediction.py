"""Recovery-prediction access for the investigation layer (Phase 3).

Consumes the existing Phase 2 artifacts without retraining or redesign: the
point-in-time features come from the unchanged ``features.build_features``
and the model is the committed ``models/baseline/model.json``. The prediction
is legitimately available before an investigation starts, so handing it to
the agent creates no leakage.
"""

import sys
from pathlib import Path

import xgboost as xgb

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import features  # noqa: E402

from agent.schemas import RecoveryPrediction  # noqa: E402

_MODEL_CACHE: dict[str, xgb.Booster] = {}
_FEATURES_CACHE: dict[str, tuple] = {}


def _features_cache_key(data_dir: str | None, frames) -> str:
    if frames is not None:
        # DB-mode frames are cached by the backend loader, so the object
        # identity is stable for the lifetime of the cached frames.
        return f"frames:{id(frames)}"
    return f"csv:{Path(data_dir or 'data').resolve()}"


def _get_features(data_dir: str | None = None, frames=None):
    key = _features_cache_key(data_dir, frames)
    if key not in _FEATURES_CACHE:
        if frames is not None:
            _FEATURES_CACHE[key] = features.build_features_from_frames(frames)
        else:
            _FEATURES_CACHE[key] = features.build_features(data_dir or "data")
    return _FEATURES_CACHE[key]


def _get_booster(model_path: str) -> xgb.Booster | None:
    path = Path(model_path)
    if not path.exists():
        return None
    key = str(path.resolve())
    if key not in _MODEL_CACHE:
        booster = xgb.Booster()
        booster.load_model(key)
        _MODEL_CACHE[key] = booster
    return _MODEL_CACHE[key]


def get_recovery_prediction(
    transaction_id: str,
    data_dir: str = "data",
    model_path: str = "models/baseline/model.json",
    frames=None,
) -> RecoveryPrediction | None:
    """Return the baseline recovery probability for a failed transaction.

    ``frames`` optionally supplies pre-loaded (customers, transactions,
    attempts, failures) frames (Phase 6 DB mode); otherwise the CSV data_dir
    is used. Returns None when the transaction is not a failed transaction
    (or does not exist) or when the model artifact is unavailable.
    """
    X, _y, meta = _get_features(data_dir, frames)
    matches = meta.index[meta["transaction_id"] == transaction_id]
    if len(matches) == 0:
        return None
    row_position = matches[0]
    booster = _get_booster(model_path)
    if booster is None:
        return None
    dmatrix = xgb.DMatrix(X.iloc[[row_position]])
    probability = float(booster.predict(dmatrix)[0])
    return RecoveryPrediction(
        transaction_id=transaction_id,
        probability=probability,
        model_path=str(Path(model_path)),
        prediction_time=meta.loc[row_position, "prediction_time"],
    )
