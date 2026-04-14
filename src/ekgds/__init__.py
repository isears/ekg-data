import abc
import numpy as np
from torch.utils.data import Dataset


class BaseEKGProcessingDS(Dataset, abc.ABC):
    """
    Abstract base dataset for processing EKG data.

    Provides a standardized interface for caching scripts. Any database-specific
    dataset (e.g., PTBXL, MIMIC) intended for offline preprocessing should inherit
    from this class and implement the required methods.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the total number of records in the dataset."""
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[np.ndarray, dict]:
        """
        Process and return the EKG signal and its metadata.

        Args:
            index (int): Index of the record.

        Returns:
            tuple[np.ndarray, dict]: A tuple containing the processed EKG signal array and a dictionary of metadata.
        """
        pass

    @staticmethod
    def postprocessing_check(sig: np.ndarray, metadata: dict) -> list[str]:
        """
        Perform sanity checks on the processed signal to ensure filtering and
        processing steps did not corrupt the data. Returns a list of warning flags.
        """
        record_id = (
            metadata.get("ecg_id")
            or metadata.get("record_name")
            or metadata.get("id", "unknown")
        )

        # 1. NaN and Inf check
        if np.isnan(sig).any() or np.isinf(sig).any():
            raise ValueError(
                f"Signal contains NaN or Inf values for record: {record_id}"
            )

        warnings_list = []

        # 2. Dead lead check (Standard deviation near 0)
        # Using axis=-1 assuming the last dimension is the time/sequence dimension
        stds = np.std(sig, axis=-1)
        if np.any(stds < 1e-6):
            warnings_list.append("dead_lead")

        # 3. Baseline wander / DC offset check
        # Means should be close to 0 if a high-pass filter was applied
        means = np.mean(sig, axis=-1)
        if np.any(np.abs(means) > 0.5):
            warnings_list.append("baseline_wander")

        return warnings_list
