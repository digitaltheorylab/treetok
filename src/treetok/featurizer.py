"""High-level convenience wrapper for dataset featurization."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .data import (
    DatasetConfig,
    build_dataset,
    feature_matrix,
    read_dataset,
    write_dataset,
)


@dataclass
class TreetokFeaturizer:
    """Lazy featurizer for labeled training data.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier
    cfg : DatasetConfig or None
        Optional default configuration for `generate`

    """

    model_name: str
    cfg: DatasetConfig | None = None
    table_: pa.Table | None = None

    def _ensure_table(self) -> pa.Table:
        """Ensure this object has a table of features."""
        if self.table_ is None:
            raise RuntimeError(
                "No dataset is loaded. Call TreetokFeaturizer.generate() or "
                ".read() first."
            )

        return self.table_

    def generate(
        self, cfg: DatasetConfig | None = None
    ) -> "TreetokFeaturizer":
        """Generate labeled training data in-place.

        Parameters
        ----------
        cfg : DatasetConfig or None
            Optional configuration overriding the instance default

        Returns
        -------
        TreetokFeaturizer
            Self with features
        """
        cfg = cfg or self.cfg
        self.table_ = build_dataset(self.model_name, cfg)

        return self

    def read(self, path: str) -> "TreetokFeaturizer":
        """Load a labeled dataset from Parquet in-place.

        Parameters
        ----------
        path : str
            Parquet file path

        Returns
        -------
        TreetokFeaturizer
            Self with features
        """
        self.table_ = read_dataset(path)

        return self

    def write(self, path: str) -> None:
        """Write the current dataset table to Parquet.

        Parameters
        ----------
        path
            Output Parquet path
        """
        write_dataset(self._ensure_table(), path)

    @property
    def table(self) -> pa.Table:
        """Return the current labeled dataset table."""
        return self._ensure_table()

    @property
    def feature_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(X, y)` extracted from the current dataset table."""
        return feature_matrix(self._ensure_table())
