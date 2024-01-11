from src.segmentation.models import *
from pathlib import Path
from src.segmentation.transforms import *
from monai.transforms import (
    Compose,
    CopyItemsd,
    KeepLargestConnectedComponentd,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    FillHolesd,
)

ckpt_dir_path: str = (Path(__file__).parent.parent.parent.resolve() / "DL_MODEL_PARAMS").as_posix()
print(ckpt_dir_path)


def compute_low_res_brainmask(
    input_image: str, output_cropped_image: str, output_brainmask: str, device
) -> None:
    print("\nComputing Low-Resolution Brainmask")
    low_res_model = LowResBrainmaskModel.load_from_checkpoint(
        checkpoint_path=f"{ckpt_dir_path}/low_res_brainmask_model.ckpt"
    )
    low_res_model.to(device)

    lr_transforms = Compose(
        [
            CopyItemsd(keys=["image"], times=1, names=["og_image"]),
            LoadITKImaged(keys=["image", "og_image"], pixel_types=[itk.F, itk.F]),
            ResampleRoughRegiond(
                keys=["image"], spacing=[3, 3, 3], size=[64, 64, 64], inference=True
            ),
            ITKImageToNumpyd(keys=["image"]),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            AddChanneld(keys=["image"]),
            ToTensord(keys=["image"], dtype=torch.float32),
        ]
    )

    # create a test dataset with the preprocessed images
    d = lr_transforms({"image": input_image})
    with torch.no_grad():
        predicted = low_res_model.model(d["image"].unsqueeze(dim=0).to(device))
        out_im = (
            torch.argmax(predicted, dim=1).detach().cpu()
        )  # convert from one hot encoding to 1 dimensional

    d["inferred_label"] = out_im
    d["inferred_label_meta_dict"] = d["image_meta_dict"]

    post_transforms = Compose(
        [
            KeepLargestConnectedComponentd(keys=["inferred_label"], applied_labels=[1]),
            FillHolesd(keys=["inferred_label"], applied_labels=[1]),
            ToITKImaged(keys=["inferred_label"]),
            ResampleMaskToOgd(keys=["inferred_label", "og_image"]),
            CropByResampled(keys=["og_image", "inferred_label"]),
            SaveITKImaged(
                keys=["og_image", "inferred_label"],
                output_filenames=[output_cropped_image, output_brainmask],
            ),
        ]
    )
    post_transforms(d)


def compute_high_res_brainmask(input_image: str, output_brainmask: str, device) -> None:
    print("\nComputing High-Resolution Brainmask")
    high_res_model = HighResBrainmaskModel.load_from_checkpoint(
        checkpoint_path=f"{ckpt_dir_path}/high_res_brainmask_model.ckpt"
    )
    high_res_model.to(device)
    hr_transforms = Compose(
        [
            CopyItemsd(keys=["image"], times=1, names=["og_image"]),
            LoadITKImaged(keys=["image", "og_image"], pixel_types=[itk.F, itk.F]),
            ITKImageToNumpyd(keys=["image"]),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            AddChanneld(keys=["image"]),
            ToTensord(keys=["image"], dtype=torch.float32),
        ]
    )
    d = hr_transforms({"image": input_image})
    with torch.no_grad():
        predicted = high_res_model.model(d["image"].unsqueeze(dim=0).to(device))
        out_im = (
            torch.argmax(predicted, dim=1).detach().cpu()
        )  # convert from one hot encoding to 1 dimensional

    d["inferred_label"] = out_im
    d["inferred_label_meta_dict"] = d["image_meta_dict"]

    post_transforms = Compose(
        [
            KeepLargestConnectedComponentd(keys=["inferred_label"], applied_labels=[1]),
            FillHolesd(keys=["inferred_label"], applied_labels=[1]),
            ToITKImaged(keys=["inferred_label"]),
            ResampleMaskToOgd(keys=["inferred_label", "og_image"]),
            SaveITKImaged(keys=["inferred_label"], output_filenames=[output_brainmask]),
        ]
    )
    post_transforms(d)


def compute_icv_mask(input_data: dict, output_mask: str, device) -> None:
    print("\nComputing Intracranial Volume mask")
    icv_model = ICVModel.load_from_checkpoint(
        checkpoint_path=f"{ckpt_dir_path}/icv_model.ckpt"
    )
    icv_model.to(device)
    icv_transforms = Compose(
        [
            CopyItemsd(keys=["t1w"], times=1, names=["og_image"]),
            LoadITKImaged(
                keys=["t1w", "t2w", "og_image", "label"],
                pixel_types=[itk.F, itk.F, itk.F, itk.UC],
            ),
            CropAndStripByResampled(
                keys=["t1w", "t2w", "label"], inference=True, skullstrip=False
            ),
            ITKImageToNumpyd(keys=["t1w", "t2w"]),
            ScaleIntensityRangePercentilesd(
                keys=["t1w", "t2w"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            CombineImagesd(keys=["t1w", "t2w"]),
            ToTensord(keys=["image"], dtype=torch.float32),
        ]
    )
    d = icv_transforms(input_data)
    with torch.no_grad():
        predicted = icv_model.model(d["image"].unsqueeze(dim=0).to(device))
        out_im = (
            torch.argmax(predicted, dim=1).detach().cpu()
        )  # convert from one hot encoding to 1 dimensional

    d["inferred_label"] = out_im
    d["inferred_label_meta_dict"] = d["t1w_meta_dict"]

    post_transforms = Compose(
        [
            FillHolesd(keys=["inferred_label"], applied_labels=[1]),
            KeepLargestConnectedComponentd(keys=["inferred_label"], applied_labels=[1]),
            ToITKImaged(keys=["inferred_label"]),
            ResampleMaskToOgd(keys=["inferred_label", "og_image"]),
            SaveITKImaged(keys=["inferred_label"], output_filenames=[output_mask]),
        ]
    )
    post_transforms(d)


def compute_gwc_mask(input_data: dict, output_mask: str, device) -> None:
    print("\nComputing Grey-White-CSF Mask")
    gwc_model = GWCModel.load_from_checkpoint(
        checkpoint_path=f"{ckpt_dir_path}/gwc_model.ckpt"
    )
    gwc_model.to(device)
    gwc_transforms = Compose(
        [
            CopyItemsd(keys=["t1w"], times=1, names=["og_image"]),
            LoadITKImaged(
                keys=["t1w", "t2w", "og_image", "icv_mask"],
                pixel_types=[itk.F, itk.F, itk.F, itk.UC],
            ),
            CropAndStripByResampled(
                keys=["t1w", "t2w", "icv_mask"], inference=True, skullstrip=True
            ),
            ITKImageToNumpyd(keys=["t1w", "t2w"]),
            ScaleIntensityRangePercentilesd(
                keys=["t1w", "t2w"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            CombineImagesd(keys=["t1w", "t2w"]),
            ToTensord(keys=["image"], dtype=torch.float32),
        ]
    )
    d = gwc_transforms(input_data)
    with torch.no_grad():
        predicted = gwc_model.model(d["image"].unsqueeze(dim=0).to(device))
        out_im = (
            torch.argmax(predicted, dim=1).detach().cpu()
        )  # convert from one hot encoding to 1 dimensional

    d["inferred_label"] = out_im
    d["inferred_label_meta_dict"] = d["t1w_meta_dict"]

    post_transforms = Compose(
        [
            FillHolesd(keys=["inferred_label"], applied_labels=[1, 2, 3]),
            ToITKImaged(keys=["inferred_label"]),
            ResampleMaskToOgd(keys=["inferred_label", "og_image"]),
            SaveITKImaged(keys=["inferred_label"], output_filenames=[output_mask]),
        ]
    )
    post_transforms(d)


def compute_seg_mask(input_data: dict, output_mask: str, device) -> None:
    print("\nComputing Segmentation Mask")
    seg_model = SegModel.load_from_checkpoint(
        checkpoint_path=f"{ckpt_dir_path}/seg_model.ckpt"
    )
    seg_model.to(device)
    seg_transforms = Compose(
        [
            CopyItemsd(keys=["t1w"], times=1, names=["og_image"]),
            LoadITKImaged(
                keys=["t1w", "t2w", "og_image", "icv_mask"],
                pixel_types=[itk.F, itk.F, itk.F, itk.UC],
            ),
            CropAndStripByResampled(
                keys=["t1w", "t2w", "icv_mask"],
                inference=True,
                skullstrip=True,
                size=[96, 128, 96],
                adjust_centroid=[0, -10, 0],
            ),
            ITKImageToNumpyd(keys=["t1w", "t2w"]),
            ScaleIntensityRangePercentilesd(
                keys=["t1w", "t2w"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            CombineImagesd(keys=["t1w", "t2w"]),
            ToTensord(keys=["image"], dtype=torch.float32),
        ]
    )
    d = seg_transforms(input_data)
    with torch.no_grad():
        predicted = seg_model.model(d["image"].unsqueeze(dim=0).to(device))
        out_im = (
            torch.argmax(predicted, dim=1).detach().cpu()
        )  # convert from one hot encoding to 1 dimensional

    d["inferred_label"] = out_im
    d["inferred_label_meta_dict"] = d["t1w_meta_dict"]

    post_transforms = Compose(
        [
            FillHolesd(keys=["inferred_label"], applied_labels=[1, 2, 3, 4]),
            KeepLargestConnectedComponentd(
                keys=["inferred_label"], applied_labels=[1, 2, 3, 4]
            ),
            ToITKImaged(keys=["inferred_label"]),
            ResampleMaskToOgd(keys=["inferred_label", "og_image"]),
            SaveITKImaged(keys=["inferred_label"], output_filenames=[output_mask]),
        ]
    )
    post_transforms(d)
