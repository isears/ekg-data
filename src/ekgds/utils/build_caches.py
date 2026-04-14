import os
import argparse
import importlib
import pkgutil
import inspect
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional

import ekgds
from ekgds import BaseEKGProcessingDS


def custom_collate(batch):
    """
    Stacks the numpy arrays into a batch but leaves the metadata dictionaries
    as a list to prevent PyTorch from attempting to cast them into tensors.
    """
    sigs = np.stack([item[0] for item in batch])
    metas = [item[1] for item in batch]
    return sigs, metas


def get_available_datasets():
    """Dynamically discovers all dataset classes inheriting from BaseEKGProcessingDS."""
    # Dynamically import all modules in the ekgds package to register subclasses
    for _, module_name, _ in pkgutil.iter_modules(ekgds.__path__, ekgds.__name__ + "."):
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue

    datasets = {}
    for cls in BaseEKGProcessingDS.__subclasses__():
        # Use the module name as the identifier (e.g. 'ptbxl' from 'ekgds.ptbxl')
        dataset_name = cls.__module__.split(".")[-1]
        datasets[dataset_name] = cls

    return datasets


def build_dataset_cache(
    dataset_cls,
    dataset_name: str,
    cache_dir: str,
    num_workers: int,
    batch_size: int,
    dataset_kwargs: Optional[dict] = None,
):
    print(f"Building {dataset_name.upper()} cache in {cache_dir}...")

    os.makedirs(cache_dir, exist_ok=True)

    memmap_path = os.path.join(cache_dir, f"{dataset_name}_signals.npy")

    dataset_kwargs = dataset_kwargs or {}
    try:
        dataset = dataset_cls(**dataset_kwargs)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return

    # Using a dataloader allows us to process the signals in parallel across multiple CPU cores
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
        shuffle=False,
    )

    all_metadata = []
    memmap_array = None

    current_idx = 0
    for i, (sig_batch, meta_batch) in enumerate(
        tqdm(
            dataloader, total=len(dataloader), desc=f"Processing {dataset_name.upper()}"
        )
    ):
        batch_sz = sig_batch.shape[0]

        # Initialize the memmap array dynamically on the first pass
        if memmap_array is None:
            memmap_array = np.lib.format.open_memmap(
                memmap_path,
                dtype=sig_batch.dtype,
                mode="w+",
                shape=(len(dataset), *sig_batch.shape[1:]),
            )

        # Write the batch chunk to the memory-mapped file
        memmap_array[current_idx : current_idx + batch_sz] = sig_batch

        # Flush periodically to free up OS RAM
        if i % max(1, (1000 // batch_size)) == 0:
            memmap_array.flush()

        for j, meta in enumerate(meta_batch):
            meta["memmap_index"] = current_idx + j
            all_metadata.append(meta)

        current_idx += batch_sz

    if memmap_array is not None:
        memmap_array.flush()

    # Save the metadata dataframe
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = os.path.join(cache_dir, f"{dataset_name}_metadata.parquet")
    metadata_df.to_parquet(metadata_path)

    print(
        f"{dataset_name.upper()} cache built successfully. Metadata saved to {metadata_path}"
    )


def main():
    available_datasets = get_available_datasets()
    dataset_choices = list(available_datasets.keys())

    # Base parser without help so Pass 1 doesn't intercept '-h'
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--dataset",
        type=str,
        choices=dataset_choices,
        required=True,
        help="Which dataset to build the cache for",
    )
    base_parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/ekgds"),
        help="Root directory to store the caches (default: ~/.cache/ekgds)",
    )
    base_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers for parallel processing (default: 4)",
    )
    base_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dataloader (default: 32)",
    )

    # PASS 1: Parse known base arguments
    args, _ = base_parser.parse_known_args()

    # PASS 2: Create main parser with help enabled, inheriting base arguments
    parser = argparse.ArgumentParser(
        parents=[base_parser], description="Build processed caches for EKG datasets."
    )

    dataset_arg_names = []
    if args.dataset and args.dataset in available_datasets:
        dataset_cls = available_datasets[args.dataset]
        sig = inspect.signature(dataset_cls.__init__)

        group = parser.add_argument_group(f"{args.dataset} specific arguments")
        for name, param in sig.parameters.items():
            if name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            dataset_arg_names.append(name)

            # Handle boolean arguments natively
            is_bool = param.annotation is bool or isinstance(param.default, bool)
            if is_bool:
                action = "store_false" if param.default is True else "store_true"
                group.add_argument(
                    f"--{name}",
                    action=action,
                    help=f"Toggle {name} (default: {param.default})",
                )
            else:
                # Infer basic types, fallback to str
                arg_type = str
                if param.annotation != inspect.Parameter.empty and isinstance(
                    param.annotation, type
                ):
                    arg_type = (
                        param.annotation
                        if param.annotation in (int, float, str)
                        else str
                    )
                elif (
                    param.default != inspect.Parameter.empty
                    and param.default is not None
                ):
                    arg_type = type(param.default)

                default_val = (
                    param.default if param.default != inspect.Parameter.empty else None
                )
                group.add_argument(
                    f"--{name}",
                    type=arg_type,
                    default=default_val,
                    help=f"(default: {default_val})",
                )

    # Now do the final, full parse. Argparse handles ALL validation and typing!
    final_args = parser.parse_args()

    # Extract just the parsed kwargs meant for the dataset
    dataset_kwargs = {
        name: getattr(final_args, name)
        for name in dataset_arg_names
        if hasattr(final_args, name) and getattr(final_args, name) is not None
    }

    if final_args.dataset in available_datasets:
        build_dataset_cache(
            available_datasets[final_args.dataset],
            final_args.dataset,
            final_args.cache_dir,
            final_args.workers,
            final_args.batch_size,
            dataset_kwargs,
        )
    else:
        print(f"Dataset '{final_args.dataset}' is not recognized or failed to load.")


if __name__ == "__main__":
    main()
