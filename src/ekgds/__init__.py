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
