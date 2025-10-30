import argparse
import logging
import multiprocessing as mp
import zipfile
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Concatenate, List, Optional, ParamSpec, TypeVar

from tqdm import tqdm

from torram.utils.config import setup_logging_config

P = ParamSpec("P")
R = TypeVar("R")


def _process_seq(
    args,
    process_function: Callable[[Path, Path, Any], List[Path]],
    **kwargs,
) -> List[Path]:
    f, dataset_dir, temp_dir = args

    relative_path_to_dir = f.parent.relative_to(dataset_dir)
    output_dir = Path(temp_dir) / relative_path_to_dir / f.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    return process_function(sequence_path=f, output_dir=output_dir, **kwargs)


def preprocess_to_zip(
    dataset_dir: Path,
    output_file: Path,
    list_sequences: Callable[[Path], List[Path]],
    process_function: Callable[Concatenate[Path, Path, P], List[Path]],
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> None:
    """Preprocess a list of sequence files and store the results in a zip file.

    List all sequence files in the dataset directory using the provided `list_sequences`
    function, which should take the dataset directory path as input and return a list
    of sequence file paths.

    def list_sequences(dataset_dir: Path) -> List[Path]:
        ...

    Processes each sequence file using the provided `process_function`, which should
    take a sequence file path and an output directory path as input, along with any
    additional keyword arguments, and return a list of paths to the processed files.

    def process_function(sequence_path: Path, output_dir: Path, **kwargs) -> List[Path]:
        ...

    @param dataset_dir: Path to the dataset directory.
    @param output_file: Path to the output zip file.
    @param list_sequences: Function to list sequence files in the dataset directory.
    @param process_function: Function to preprocess each sequence file.
    @param cache_dir: Optional path to a cache directory for intermediate files.
    @param kwargs: Additional keyword arguments to pass to the preprocessing function.
    """

    assert output_file.suffix == ".zip", f"Output file must be a zip file, got {output_file}"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)

    if cache_dir is not None:
        temp_dir = cache_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        cache_dir_path = temp_dir
    else:
        temp_dir = TemporaryDirectory()
        cache_dir_path = Path(temp_dir.name)

    # List all sequence files in the dataset directory.
    sequences = list_sequences(dataset_dir)

    # Process the files in parallel and store the results in the cache directory.
    logger.info(f"Processing {len(sequences)} sequences")
    args = [(f, dataset_dir, cache_dir_path) for f in sequences]
    func = partial(_process_seq, process_function=process_function, **kwargs)

    processed = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        processed += list(tqdm(pool.imap_unordered(func, args), total=len(args)))
    processed = sum(processed, [])  # Flatten the list of lists.

    # Write the processed files to a zip file.
    logger.info(f"Writing processed sequences to zip file {output_file}")
    with zipfile.ZipFile(output_file, "w") as writer:
        for pf in tqdm(processed, desc="Writing zip"):
            relative_path = pf.relative_to(cache_dir_path)
            writer.write(pf, arcname=relative_path)

    # Clean up the temporary directory if not using a cache directory.
    if isinstance(temp_dir, TemporaryDirectory):
        temp_dir.cleanup()


def preprocess_main(
    list_sequences: Callable[[Path], List[Path]],
    process_function: Callable[Concatenate[Path, Path, P], List[Path]],
    parser: Optional[argparse.ArgumentParser] = None,
):
    """Main function to preprocess a dataset and store the results in a zip file.

    @param list_sequences: List sequence function (see preprocess_to_zip).
    @param process_function: Preprocessing function (see preprocess_to_zip).
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Setup logging and print config.
    setup_logging_config(debug=args.debug)
    logger = logging.getLogger(__name__)
    kwargs = {
        key: value
        for key, value in vars(args).items()
        if key not in ["dataset_dir", "output_file", "cache_dir", "debug"]
    }
    logger.info(f"Preprocessing with args: {vars(args)}")

    # Run preprocessing main function in multi-process mode.
    preprocess_to_zip(
        dataset_dir=args.dataset_dir,
        output_file=args.output_file,
        list_sequences=list_sequences,
        process_function=process_function,
        cache_dir=args.cache_dir,
        **kwargs,
    )
