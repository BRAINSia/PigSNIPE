import hydra
from omegaconf import OmegaConf
import torch
from src.landmarks.common import run_episode
from src.landmarks.transforms import *
from monai.transforms import (
    Compose,
    ScaleIntensityRangePercentilesd,
    CastToTyped,
)
import numpy as np

PANDAS_FCSV_KWARGS = {
    "comment": "#",
    "names": [
        "id",
        "x",
        "y",
        "z",
        "ow",
        "ox",
        "oy",
        "oz",
        "vis",
        "sel",
        "lock",
        "label",
        "desc",
        "associatedNodeID",
    ],
}


def get_centroid_from_brainmask(brainmask):
    print(f"in get centroid from brainmask, path: {brainmask}")
    brainmask_itk = itk.imread(brainmask, itk.UC)
    moments_calc = itk.ImageMomentsCalculator[type(brainmask_itk)].New()
    moments_calc.SetImage(brainmask_itk)
    moments_calc.Compute()
    centroid = moments_calc.GetCenterOfGravity()
    print(f"centroid: {centroid}")

    return list(centroid)


def get_prim1_transforms():
    tfms = Compose(
        [
            LoadITKImaged(keys=["image"]),
            ITKImageToNumpyd(keys=["image"]),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=10.0,
                upper=99.0,
                b_min=0.0,
                b_max=255.0,
                clip=True,
                relative=False,
            ),
            # cast to uint8
            CastToTyped(keys=["image"], dtype=np.uint8),
            ToITKImaged(keys=["image"]),
        ]
    )
    return tfms


def get_prim2_transforms(landmarks):
    tfms = Compose(
        [
            LoadITKImaged(keys=["image"]),
            ITKImageToNumpyd(keys=["image"]),
            LoadCSVd(keys=["prior"], pd_kwargs=PANDAS_FCSV_KWARGS),
            ExtractLandmarks2d(keys=["prior"], landmark_names=landmarks),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=10.0,
                upper=99.0,
                b_min=0.0,
                b_max=255.0,
                clip=True,
                relative=False,
            ),
            # cast to uint8
            CastToTyped(keys=["image"], dtype=np.uint8),
            ToITKImaged(keys=["image"]),
        ]
    )

    return tfms


def get_ACPC_transforms():
    tfms = Compose(
        [
            LoadITKImaged(keys=["image"]),
            ITKImageToNumpyd(keys=["image"]),
            LoadCSVd(keys=["fcsv"], pd_kwargs=PANDAS_FCSV_KWARGS),
            ExtractLandmarksForTransformd(keys=["fcsv"]),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=10.0,
                upper=99.0,
                b_min=0.0,
                b_max=255.0,
                clip=True,
                relative=False,
            ),
            # cast to uint8
            CastToTyped(keys=["image"], dtype=np.uint8),
            ToITKImaged(keys=["image"]),
        ]
    )

    return tfms


def inference_on_one(
    config,
    model,
    stage,
    input_im_path,
    output_filename,
    input_fcsv_path=None,
    centroid=None,
    gpu_id=0,
    write_lps=False,
):
    # note: you can import and use this function anywhere you have these four file paths (config, model, input_filename, output_filename)
    config = OmegaConf.load(config)
    gpu_id = int(gpu_id)
    config.model.gpu = gpu_id
    config.run_mode = "inf"

    test_env = hydra.utils.instantiate(config.environment)
    landmarks = config.environment.landmarks

    if stage == "prim1":
        assert (
            centroid is not None
        ), "First stage of primary landmarks detection requires centroid of the brain"
        episode_data = get_prim1_transforms()({"image": input_im_path})
        episode_data["otsu_center"] = centroid

    elif stage == "prim2":
        assert (
            input_fcsv_path is not None
        ), "Second stage of primary landmarks detection requires fiducial file from the first stage"
        episode_data = get_prim2_transforms(landmarks)(
            {"image": input_im_path, "prior": input_fcsv_path}
        )
    elif stage == "prim3":
        assert (
            input_fcsv_path is not None
        ), "third stage of primary landmarks detection requires fiducial file from the first stage"
        episode_data = get_prim2_transforms(landmarks)(
            {"image": input_im_path, "fcsv": input_fcsv_path}
        )
    elif stage == "ACPCsec" or stage == "ACPCter":
        assert (
            input_fcsv_path is not None
        ), "Landmark detection in ACPC space requires primary landmarks fiducial file"
        episode_data = get_ACPC_transforms()(
            {"image": input_im_path, "fcsv": input_fcsv_path}
        )
    else:
        raise ValueError("Invalid stage")

    policy = hydra.utils.instantiate(config.model)
    policy.load_state_dict(torch.load(model, map_location=policy.device))
    policy.to(policy.device)
    policy.eval()

    _, final_infos, _ = run_episode(
        env=test_env,
        policy_network=policy,
        eps=0.0,
        episode_data=episode_data,
    )

    with open(f"{output_filename}", "w") as output_fcsv_file:
        output_fcsv_file.write(
            "# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )
        for agend_idx, agent_id in enumerate(final_infos.keys()):
            final_coord = final_infos[agent_id]["physical_space_location"]
            if not write_lps:
                # flip to ras
                final_coord = np.array(final_coord) * np.array([-1.0, -1.0, 1.0])

            output_fcsv_file.write(
                "vtkMRMLMarkupsFiducialNode_{},{},{},{},0,0,0,1,1,1,0,{},,\n".format(
                    agend_idx, final_coord[0], final_coord[1], final_coord[2], agent_id
                )
            )
