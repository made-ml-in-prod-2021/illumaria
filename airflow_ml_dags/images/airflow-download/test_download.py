from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from download import download


@pytest.fixture()
def n_samples() -> int:
    return 150


@pytest.fixture
def n_features() -> int:
    return 4


def test_download(n_samples: int, n_features: int) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        save_path = Path("data", "raw")
        result = runner.invoke(
            download,
            [
                str(save_path),
            ],
        )
        assert result.exit_code == 0
        features_df = pd.read_csv(Path(save_path, "data.csv"))
        assert type(features_df) == pd.DataFrame
        assert features_df.shape == (n_samples, n_features)
        targets_series = pd.read_csv(
            Path(save_path, "target.csv"), squeeze=True, dtype=int
        )
        assert type(targets_series) == pd.Series
        assert targets_series.shape == (n_samples,)
        assert targets_series.dtype == int
        assert targets_series.min() >= 0
        assert targets_series.max() <= 2
