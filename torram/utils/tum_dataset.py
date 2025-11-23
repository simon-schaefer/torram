import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from PIL import Image
from scipy.spatial.transform import Rotation

TUM_DEPTH_SCALE = 5000.0  # Depth images are scaled by this factor.


class TUMDatasetWriter:
    """Class to write TUM RGB-D dataset format files (iteratively)."""

    def __init__(self, directory: Path, clean_if_exists: bool = True):
        """Initialize TUM dataset writer.

        @param directory: Directory to write dataset files to.
        @param clean_if_exists: If True, remove existing directory before writing.
        """
        if clean_if_exists and directory.exists():
            shutil.rmtree(directory)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_transform(
        self,
        name: str,
        timestamp_ns: int,
        transform: Union[Float[torch.Tensor, "4 4"], Float[np.ndarray, "4 4"]],
    ):
        """Write a transform to a TUM format trajectory file.

        @param name: Name of the trajectory file (without extension).
        @param timestamp_ns: Timestamp in nanoseconds.
        @param transform: 4x4 transformation matrix.
        """
        file_path = self.directory / f"{name}.txt"
        with open(file_path, "a") as f:
            t = " ".join(map(str, transform[:3, 3].tolist()))
            q = " ".join(map(str, Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()))
            f.write(f"{timestamp_ns} {t} {q}\n")

    def write_associations_file(self, name: str, sensor_1: str, sensor_2: str):
        """Write an associations file between two sensors (e.g., RGB and depth).

        @param name: Name of the associations file (without extension).
        @param sensor_1: Name of the first sensor file (without extension).
        @param sensor_2: Name of the second sensor file (without extension).
        """
        assoc_file = self.directory / f"{name}.txt"
        sensor_1_fn = self.directory / sensor_1 / "data.txt"
        sensor_2_fn = self.directory / sensor_2 / "data.txt"

        df1 = pd.read_csv(sensor_1_fn, sep="\s+", header=0)
        df2 = pd.read_csv(sensor_2_fn, sep="\s+", header=0)
        assoc_df = pd.merge(df1, df2, on="#", how="inner", suffixes=(f"_1", f"_2"))

        with open(assoc_file, "w") as f:
            for _, row in assoc_df.iterrows():
                # column names shifted due to "# timestamp filename" (space separated)
                fn_1 = row[f"timestamp_1"]
                fn_2 = row[f"timestamp_2"]
                ts = row["#"]
                f.write(f"{ts} {fn_1} {ts} {fn_2}\n")

    def write_image(
        self,
        name: str,
        timestamp_ns: int,
        image: Union[np.ndarray, Path, str, torch.Tensor],
    ) -> Path:
        """Write an image to the TUM dataset format.

        @param name: Name of the sensor.
        @param timestamp_ns: Timestamp in nanoseconds.
        @param image: Image as an array or path to an existing image file.
        @return: Path to the written image file.
        """
        # Write image file.
        image_dir = self.directory / name / "data"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_filename = image_dir / f"{timestamp_ns}.png"
        if isinstance(image, np.ndarray):
            image_ = Image.fromarray(image)
            image_.save(image_filename)
        elif isinstance(image, Path):
            os.symlink(image.as_posix(), image_filename)
        elif isinstance(image, str):
            os.symlink(image, image_filename)
        elif isinstance(image, torch.Tensor):
            image_ = Image.fromarray(image.cpu().numpy())
            image_.save(image_filename)
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Unknown image type: {type(image)}, skipping write.")
            return

        # Append to listing file if not skipped.
        listing_file = self.directory / name / "data.txt"
        if not listing_file.exists():
            with open(listing_file, "w") as f:
                f.write("# timestamp filename\n")

        with open(listing_file, "a") as f:
            image_fn_relative = image_filename.relative_to(self.directory)
            f.write(f"{timestamp_ns} {image_fn_relative}\n")

        return image_filename

    def write_depth_image(
        self,
        name: str,
        timestamp_ns: int,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Path:
        """Write a depth image to the TUM dataset format.

        @param name: Name of the sensor.
        @param timestamp_ns: Timestamp in nanoseconds.
        @param image: Image as array in meters.
        @return: Path to the written depth image file.
        """
        if isinstance(image, torch.Tensor):
            image_np = (image.cpu().numpy() * TUM_DEPTH_SCALE).astype(np.uint16)
        else:
            image_np = (image * TUM_DEPTH_SCALE).astype(np.uint16)
        return self.write_image(name, timestamp_ns, image_np)

    def write_file(self, filename: str, src: Union[Path, str]):
        """Write a generic file to the TUM dataset directory.

        @param filename: Name of the file to write (with extension).
        @param src: Source file path or content as string.
        """
        dest_file = self.directory / filename
        shutil.copy(src, dest_file)
