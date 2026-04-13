import os
import torch
import pandas as pd
import numpy as np
import ast
import wfdb
import neurokit2 as nk
from typing import Optional


def load_single_ptbxl_record(
    root_dir: str, id: int, lowres: bool
) -> tuple[np.ndarray, dict]:
    id_str = "{:05d}".format(id)
    prefix_dir = f"{id_str[:2]}000"

    if lowres:
        return wfdb.rdsamp(f"{root_dir}/records100/{prefix_dir}/{id_str}_lr")  # type: ignore
    else:
        return wfdb.rdsamp(f"{root_dir}/records500/{prefix_dir}/{id_str}_hr")  # type: ignore


class PtbxlProcessingDS(torch.utils.data.Dataset):

    def __init__(
        self,
        root_folder: Optional[str] = None,
        lowres: bool = False,
    ):
        """Base PTBXL dataset initialization

        Args:
            root_folder (str, optional): Path to PTBXL data. Defaults to None.
            lowres (bool, optional): Whether to use the 100Hz (True) or 500Hz (False) data. Defaults to False.

        Returns:
            None: None
        """
        super(PtbxlProcessingDS, self).__init__()

        resolved_root = root_folder or os.environ.get("PTBXL_DATA_DIR")
        if not resolved_root or not os.path.exists(resolved_root):
            raise ValueError(
                "Dataset root folder must be specified via the `root_folder` "
                "argument or the `PTBXL_DATA_DIR` environment variable, and the path must exist."
            )
        self.root_folder: str = resolved_root

        metadata = pd.read_csv(f"{self.root_folder}/ptbxl_database.csv")
        metadata = metadata.astype({"ecg_id": int, "patient_id": int})
        self.metadata = metadata

        self.lowres = lowres

        # Get PTBXL labels
        self.metadata["scp_codes"] = self.metadata.scp_codes.apply(ast.literal_eval)

        # Modified from physionet example.py
        scp_codes = pd.read_csv(f"{self.root_folder}/scp_statements.csv", index_col=0)
        scp_codes = scp_codes[scp_codes.diagnostic == 1]

        self.ordered_labels = list()

        for diagnostic_code, description in zip(scp_codes.index, scp_codes.description):
            self.ordered_labels.append(description)
            self.metadata[description] = self.metadata.scp_codes.apply(
                lambda x: diagnostic_code in x.keys()
            ).astype(float)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict]:
        # Outputs ECG data of shape sig_len x num_leads (e.g. for low res 1000 x 12)
        this_metadata = self.metadata.iloc[index].to_dict()

        sig, sigmeta = load_single_ptbxl_record(
            id=this_metadata["ecg_id"], lowres=self.lowres, root_dir=self.root_folder
        )

        sig_clean = np.apply_along_axis(
            nk.ecg_clean,
            1,
            sig.transpose(),
            sampling_rate=sigmeta["fs"],
        )  # type: ignore

        for k, v in sigmeta.items():
            assert k not in this_metadata

            if k == "units":
                # Make sure everything milivolts
                assert all([u == "mV" for u in v])

                this_metadata["units"] = "mV"

            elif k == "comments":
                this_metadata["comments"] = "\n".join(v)

            elif k == "sig_name":
                # Make sure channels in correct order
                assert v == [
                    "I",
                    "II",
                    "III",
                    "aVR",
                    "aVL",
                    "aVF",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6",
                ]

            else:
                this_metadata[k] = v

        return sig_clean, this_metadata


if __name__ == "__main__":
    ds = PtbxlProcessingDS(lowres=True)
    print(ds[0])
