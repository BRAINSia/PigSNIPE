import argparse
import glob
from pathlib import Path
import shutil

current_file_path = Path(__file__).resolve()

parser = argparse.ArgumentParser(description="Run this set up script to configure the project.")

parser.add_argument("--brains_path", "-b", type=str, default=None, help="Path to the BRAINSTools build directory.")

args = parser.parse_args()

if args.brains_path is None:
    raise ValueError("Please provide a path to the BRAINSTools build directory.")

brains_path = Path(args.brains_path)

if not brains_path.exists():
    raise ValueError(f"BRAINSTools path {brains_path} does not exist.")

brains_path = Path(glob.glob(brains_path.as_posix() + "/BRAINSTools-Release-5*")[0])

base_dir = current_file_path.parent
brains_binaries_dir = base_dir / "BRAINSToolsBinaries"

if not brains_binaries_dir.exists():
    brains_binaries_dir.mkdir()

shutil.copy(brains_path / "bin" / "BRAINSFit", brains_binaries_dir)
shutil.copy(brains_path / "bin" / "BRAINSResample", brains_binaries_dir)
shutil.copy(brains_path / "bin" / "BRAINSConstellationLandmarksTransform", brains_binaries_dir)
shutil.copy(brains_path / "bin" / "landmarksConstellationAligner", brains_binaries_dir)
shutil.copy(brains_path / "lib" / "libBRAINSFitLib.so", brains_binaries_dir)
shutil.copy(brains_path / "lib" / "libBRAINSResampleLib.so", brains_binaries_dir)
shutil.copy(brains_path / "lib" / "libBRAINSConstellationLandmarksTransformLib.so", brains_binaries_dir)
shutil.copy(brains_path / "lib" / "liblandmarksConstellationAlignerLib.so", brains_binaries_dir)
