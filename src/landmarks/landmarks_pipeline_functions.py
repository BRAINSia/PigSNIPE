from src.landmarks.transforms import *
from pathlib import Path
import itk
import sys
from monai.transforms import Compose
from src.landmarks.inference import inference_on_one
from subprocess import run

itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)


def sub_ses_from_filename(filename):
    fname = str(Path(str(filename)).name)
    result = "_".join(fname.split("_")[:2])
    return result


def merge_landmarks(lmk_path1, lmk_path2, out_path):
    with open(lmk_path1, "r") as f:
        file_lines = f.readlines()
    labels1 = []
    for line in file_lines:
        if "#" not in line:
            labels1.append(line.split(",")[11].upper())

    with open(lmk_path2, "r") as f:
        file_lines2 = f.readlines()

    lmk_lines = []
    for line in file_lines2:
        if "#" not in line:
            label = line.split(",")[11].upper()
            if label not in labels1:
                lmk_lines.append(line)

    new_lines = file_lines + lmk_lines
    with open(out_path, "w") as f:
        f.writelines(new_lines)


def create_ACPC_fcsv_and_transform(
    fcsv_filename, output_fcsv_filename, output_transform_filename
):
    command = (
        f"landmarksConstellationAligner "
        f"--inputLandmarksPaired {fcsv_filename} "
        f"--outputLandmarksPaired {output_fcsv_filename} "
        f"--outputTransform {output_transform_filename} "
    )
    run(command.split(" "), check=True)


def create_ACPC_t1w(t1w_input_filename, transform_filename, t1w_output_filename):
    command = (
        f"BRAINSResample "
        f"--inputVolume {t1w_input_filename} "
        f"--outputVolume {t1w_output_filename} "
        f"--warpTransform {transform_filename} "
        f"--pixelType float "
        f"--interpolationMode ResampleInPlace "
        f"--numberOfThreads -1"
    )
    run(command.split(" "), check=True)


def generate_cropped_image(image, brainmask, out_filename):
    transforms = Compose(
        [
            LoadITKImaged(keys=["image", "brainmask"]),
            CropByResampled(keys=["image", "brainmask"]),
            SaveITKImaged(keys=["image"], output_filename=out_filename),
        ]
    )
    data = [{"image": image, "brainmask": brainmask}]
    transforms(data)


def get_centroid_from_brainmask(brainmask):
    print("Computing centroid.\n")
    brainmask_itk = itk.imread(brainmask, itk.UC)
    moments_calc = itk.ImageMomentsCalculator[type(brainmask_itk)].New()
    moments_calc.SetImage(brainmask_itk)
    moments_calc.Compute()
    centroid = moments_calc.GetCenterOfGravity()
    print(f"centroid: {centroid}")

    return list(centroid)


def transform_landmarks_to_og(input_lmk, transform, out_path):
    command = (
        f"BRAINSConstellationLandmarksTransform "
        f"-i {input_lmk} -o {out_path} -t {transform}"
    )
    run(command.split(" "), check=True)


def run_RL_model(
    input_filename, stage, out_fname, input_fcsv=None, centroid=None, gpu=None
):
    assert stage in [
        "prim1",
        "prim2",
        "prim3",
        "ACPCsec",
        "ACPCter",
    ], "<stage> must be one of ['prim1', 'prim2', 'ACPCsec', 'ACPCter']"
    param_dir: str = (Path(__file__).parent.resolve() / "RL_Params").as_posix()
    if stage == "prim1":
        assert centroid is not None, "stage prim1 requires brainmask file"
        config_path = f"{param_dir}/prim_1mm/config.yaml"
        model_path = f"{param_dir}/prim_1mm/model.pt"
        inference_on_one(
            config=config_path,
            model=model_path,
            stage=stage,
            input_im_path=input_filename,
            output_filename=out_fname,
            centroid=centroid,
            gpu_id=gpu,
        )

    else:
        if stage == "prim2":
            config_path = f"{param_dir}/prim_05mm/config.yaml"
            model_path = f"{param_dir}/prim_05mm/model.pt"
        elif stage == "ACPCsec":
            config_path = f"{param_dir}/sec/config.yaml"
            model_path = f"{param_dir}/sec/model.pt"
        elif stage == "ACPCter":
            config_path = f"{param_dir}/ter/config.yaml"
            model_path = f"{param_dir}/ter/model.pt"

        inference_on_one(
            config=config_path,
            model=model_path,
            stage=stage,
            input_im_path=input_filename,
            output_filename=out_fname,
            input_fcsv_path=input_fcsv,
            gpu_id=gpu,
        )
