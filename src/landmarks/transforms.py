import itk
import numpy as np
import pandas as pd
from monai.transforms.transform import MapTransform, Transform
import json
from pathlib import Path


# these are my custom monai style transforms


class SaveITKImaged(object):
    def __init__(self, keys, output_filename=None, out_dir=None, output_postfix=None):
        self.keys = keys
        self.postfix = output_postfix
        self.out_dir = out_dir
        self.out_filename = output_filename

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if self.out_filename is None:
                input_filename = Path(d[f"{k}_meta_dict"]["filename"]).absolute()
                parent_dir = input_filename.parent
                basename = str(input_filename.name).split(".")[0]
                extension = ".".join(str(input_filename).split(".")[-2:])
                output_filename = (
                    f"{self.out_dir}/{basename}_{self.postfix}.{extension}"
                )
            else:
                output_filename = self.out_filename
            d["output_filename"] = output_filename
            print("writing to", output_filename)
            itk.imwrite(d[k], output_filename)
            pass

        return d


class CropByResampled(object):
    def __init__(self, keys):
        # assert len(keys) == 2, "must pass in a t1w key and label key"
        self.t1w_key = keys[0]
        self.mask_key = keys[1]

    def __call__(self, data):
        d = dict(data)
        t1w_itk_image = d[self.t1w_key]
        mask_itk_image = d[self.mask_key]

        ref_image = self.setup_reference_image(mask_itk_image, pixel_type=itk.D)

        image_type = itk.Image[itk.F, 3]
        label_type = itk.Image[itk.UC, 3]
        linear_interpolator = itk.LinearInterpolateImageFunction[
            image_type, itk.D
        ].New()
        nearest_interpolator = itk.NearestNeighborInterpolateImageFunction[
            label_type, itk.D
        ].New()
        identity_transform = itk.IdentityTransform[itk.D, 3].New()

        t1w_resampler = itk.ResampleImageFilter[image_type, image_type].New()
        t1w_resampler.SetInterpolator(linear_interpolator)
        t1w_resampler.SetTransform(identity_transform)
        t1w_resampler.SetInput(t1w_itk_image)
        t1w_resampler.SetReferenceImage(ref_image)
        t1w_resampler.UseReferenceImageOn()
        t1w_resampler.UpdateLargestPossibleRegion()

        # update the dictionary
        d[self.t1w_key] = t1w_resampler.GetOutput()
        return d

    def setup_reference_image(self, rough_label, pixel_type):
        filter = itk.LabelStatisticsImageFilter[
            type(rough_label), type(rough_label)
        ].New()
        filter.SetInput(rough_label)
        filter.SetLabelInput(rough_label)
        filter.Update()

        box = filter.GetBoundingBox(1)
        min_index = [box[0], box[2], box[4]]
        max_index = [box[1], box[3], box[5]]
        min_location = rough_label.TransformIndexToPhysicalPoint(min_index)
        max_location = rough_label.TransformIndexToPhysicalPoint(max_index)

        box_center = (np.array(min_location) + np.array(max_location)) / 2

        spacing = [0.5, 0.5, 0.5]
        origin = np.array(box_center) - (np.array([71, 95, 95]) * np.array(spacing))
        ImageType = itk.Image[pixel_type, 3]
        fixed_field = ImageType.New()
        fixed_field.SetOrigin(origin)
        # spacing = [80/144, 120/144, 120/144]

        fixed_field.SetSpacing(spacing)

        start = itk.Index[3]()
        start[0] = 0  # first index on X
        start[1] = 0  # first index on Y
        start[2] = 0  # first index on Z

        size = itk.Size[3]()
        size[0] = 144  # size along X
        size[1] = 192  # size along Y
        size[2] = 192  # size along Z

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        region.SetIndex(start)

        fixed_field.SetRegions(region)
        fixed_field.Allocate()

        return fixed_field


def get_inverse_transform(priors: dict, first_stage: dict):
    point_type = itk.Point[itk.D, 3]
    fixed_landmarks = [
        point_type(first_stage["AC"]),
        point_type(first_stage["PC"]),
        point_type(first_stage["RP"]),
    ]
    moving_landmarks = [
        point_type(priors["AC"]),
        point_type(priors["PC"]),
        point_type(priors["RP"]),
    ]
    transform = itk.VersorRigid3DTransform[itk.D].New()
    t_initializer = itk.LandmarkBasedTransformInitializer[
        itk.Transform[itk.D, 3, 3]
    ].New()
    t_initializer.SetTransform(transform)
    t_initializer.SetFixedLandmarks(fixed_landmarks)
    t_initializer.SetMovingLandmarks(moving_landmarks)
    t_initializer.InitializeTransform()
    inverse = transform.GetInverseTransform()

    return inverse


def transform_points(
    priors: dict, first_stage: dict, landmarks_to_transform, inverse_transform
):
    point_type = itk.Point[itk.D, 3]
    landmarks = first_stage
    for key in landmarks_to_transform:
        # if key not in landmarks.keys():
        landmarks[key] = np.array(
            inverse_transform.TransformPoint(point_type(priors[key]))
        )

    return landmarks


class TransformLandmarksd(MapTransform):
    def __init__(self, keys, landmarks_to_transform):
        self.prior_key = keys[0]
        self.first_stage_key = keys[1]
        self.landmarks_to_transform = landmarks_to_transform

    def __call__(self, data):
        d = dict(data)
        priors = d[self.prior_key]
        first_stage = d[self.first_stage_key]
        if self.landmarks_to_transform is None:
            d["landmarks"] = first_stage
        else:
            t_inv = get_inverse_transform(priors, first_stage)
            transformed_landmarks = transform_points(
                priors, first_stage, self.landmarks_to_transform, t_inv
            )
            d["landmarks"] = transformed_landmarks

        return d


class LoadCSV(Transform):
    def __init__(self, pd_kwargs: dict):
        super(object, self).__init__()
        self.pd_kwargs = pd_kwargs

    def __call__(self, fcsv_filename: str):
        return pd.read_csv(fcsv_filename, **self.pd_kwargs)


class LoadCSVd(MapTransform):
    def __init__(self, keys, pd_kwargs: dict):
        super(object, self).__init__()
        self.keys = keys
        self.pd_kwargs = pd_kwargs
        self.loader = LoadCSV(self.pd_kwargs)

    def __call__(self, data):
        data_dict = dict(data)
        for key in self.keys:
            data_dict[key] = self.loader(data_dict[key])
        return data_dict


def get_direction_cos_from_image(image):
    DIMS = len(image.GetOrigin())
    arr = np.array([[0.0] * DIMS] * DIMS)
    mat = image.GetDirection().GetVnlMatrix()
    for i in range(DIMS):
        for j in range(DIMS):
            arr[i][j] = mat.get(i, j)
    return arr


class ITKImageToNumpyd(MapTransform):
    def __init__(self, keys):
        self.keys = keys
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        d = self.meta_updater(d)
        for k in self.keys:
            d[k] = itk.array_from_image(d[k])

        return d


class ExtractLandmarksd(MapTransform):
    def __init__(self, keys, landmark_names, ras_to_lps=True):
        self.keys = keys
        self.landmark_names = landmark_names
        self.ras_to_lps = ras_to_lps

    def __call__(self, data):
        d = dict(data)

        for i in ["AC", "PC", "RP"]:
            if i not in self.landmark_names:
                self.landmark_names.append(i)

        for k in self.keys:
            df = d[k]
            df["label"] = df["label"].str.upper()
            output_dict = {}
            for lmk_name in self.landmark_names:
                if lmk_name in [
                    "BPONS",
                    "BASION",
                    "OPISTHION",
                    "VN4",
                    "GENU",
                    "ROSTRUM",
                    "MID_PRIM_SUP",
                    "MID_PRIM_INF",
                    "L_CAUD_HEAD",
                    "R_CAUD_HEAD",
                    "R_FRONT_POLE",
                    "L_FRONT_POLE",
                    "R_LAT_EXT",
                    "L_LAT_EXT",
                    "L_SUP_EXT",
                    "R_SUP_EXT",
                    "R_FRONT_POLE",
                    "L_FRONT_POLE",
                    "R_LAT_EXT",
                    "L_LAT_EXT",
                    "L_SUP_EXT",
                    "R_SUP_EXT",
                ]:
                    output_dict[lmk_name] = np.array([0, 0, 0])
                else:
                    try:
                        output_dict[lmk_name] = np.array(
                            [
                                df[df["label"] == lmk_name][val].values[0]
                                for val in ["x", "y", "z"]
                            ]
                        )
                        if self.ras_to_lps:
                            output_dict[lmk_name] = output_dict[lmk_name] * np.array(
                                [-1.0, -1.0, 1.0]
                            )
                    except Exception as e:
                        fail_id = d["id"]
                        print(f"FAILURE ({fail_id}) ({lmk_name})")
                        print(df)
                        raise (e)
            d[k] = output_dict
        return d


class ExtractLandmarks2d(MapTransform):
    def __init__(self, keys, landmark_names, ras_to_lps=True):
        self.keys = keys
        self.landmark_names = landmark_names
        self.ras_to_lps = ras_to_lps

    def __call__(self, data):
        d = dict(data)

        for k in self.keys:
            df = d[k]
            df["label"] = df["label"].str.upper()
            output_dict = {}
            for lmk_name in self.landmark_names:
                try:
                    output_dict[lmk_name] = np.array(
                        [
                            df[df["label"] == lmk_name][val].values[0]
                            for val in ["x", "y", "z"]
                        ]
                    )
                    if self.ras_to_lps:
                        output_dict[lmk_name] = output_dict[lmk_name] * np.array(
                            [-1.0, -1.0, 1.0]
                        )
                except Exception as e:
                    fail_id = d["id"]
                    print(f"FAILURE ({fail_id}) ({lmk_name})")
                    print(df)
                    raise (e)
            d[k] = output_dict
        return d


class ExtractLandmarksForTransformd(MapTransform):
    def __init__(self, keys, ras_to_lps=True):
        self.keys = keys
        self.ras_to_lps = ras_to_lps

    def __call__(self, data):
        d = dict(data)

        for k in self.keys:
            df = d[k]
            df["label"] = df["label"].str.upper()
            output_dict = {}
            for lmk_name in ["AC", "PC", "RP"]:
                try:
                    output_dict[lmk_name] = np.array(
                        [
                            df[df["label"] == lmk_name][val].values[0]
                            for val in ["x", "y", "z"]
                        ]
                    )
                    if self.ras_to_lps:
                        output_dict[lmk_name] = output_dict[lmk_name] * np.array(
                            [-1.0, -1.0, 1.0]
                        )
                except Exception as e:
                    fail_id = d["id"]
                    print(f"FAILURE ({fail_id}) ({lmk_name})")
                    print(df)
                    raise (e)
            d["Primary_lmks"] = output_dict
        return d


class OtsuCenter(Transform):
    def __init__(self):
        # type defs
        self.label_map_type = itk.LabelMap[itk.StatisticsLabelObject[itk.UL, 3]]
        self.image_type = itk.Image[itk.F, 3]
        self.label_type = itk.Image[itk.US, 3]

        # define filters
        self.otsu_filter = itk.OtsuThresholdImageFilter[
            self.image_type, self.label_type
        ].New()
        self.shape_labelmap_filter = itk.LabelImageToShapeLabelMapFilter[
            self.label_type, self.label_map_type
        ].New()
        # define all params
        self.shape_labelmap_filter.SetInput(self.otsu_filter.GetOutput())

    def __call__(self, itk_image):
        self.otsu_filter.SetInput(itk_image)
        self.shape_labelmap_filter.UpdateLargestPossibleRegion()
        return self.shape_labelmap_filter.GetOutput().GetNthLabelObject(0).GetCentroid()


class OtsuCenterd(MapTransform):
    def __init__(self, keys):
        self.keys = keys
        self.otsu_transform = OtsuCenter()

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = self.otsu_transform(d[k])
        return d


class GetCentroidFromJsond(MapTransform):
    def __init__(self, keys, path_to_json: str):
        self.keys = keys
        with open(path_to_json, "r") as f:
            data_json = json.load(f)
        self.centroid_data = data_json["data"]

    def sub_ses_from_filename(self, filename):
        fname = str(Path(str(filename)).name)
        result = "_".join(fname.split("_")[:2])
        return result

    def __call__(self, data):
        d = dict(data)
        image_key = self.keys[0]
        filename = self.sub_ses_from_filename(d[f"{image_key}_meta_dict"]["filename"])
        centroid = None
        for i in self.centroid_data:
            if i["id"] == filename:
                centroid = i["centroid"]
                break
        if centroid is None:
            raise ValueError("ERROR, centroid not found", filename)
        d["otsu_center"] = np.array(centroid)

        return d


class LoadITKImaged(MapTransform):
    def __init__(self, keys, pixel_type=itk.F):
        self.keys = keys
        self.pixel_type = pixel_type
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if f"{k}_meta_dict" not in d.keys():
                d[f"{k}_meta_dict"] = {"filename": d[k]}
            else:
                d[f"{k}_meta_dict"]["filename"] = d[k]

            if k == "label_96" or k == "label" or k == "mask" or k == "brainmask":
                d[k] = itk.imread(d[k], itk.UC)
            else:
                d[k] = itk.imread(d[k], self.pixel_type)

        d = self.meta_updater(d)

        return d


class UpdateMetaDatad(MapTransform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            image = d[k]
            if f"{k}_meta_dict" not in d.keys():
                d[f"{k}_meta_dict"] = {}
            d[f"{k}_meta_dict"]["origin"] = np.array(image.GetOrigin())
            d[f"{k}_meta_dict"]["spacing"] = np.array(image.GetSpacing())
            d[f"{k}_meta_dict"]["direction"] = get_direction_cos_from_image(image)

        return d


class ToITKImaged(MapTransform):
    def __init__(self, keys):
        self.keys = keys
        pass

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            meta_data = d[f"{k}_meta_dict"]
            itk_image = itk.image_from_array(d[k])
            itk_image.SetOrigin(meta_data["origin"])
            itk_image.SetSpacing(meta_data["spacing"])
            itk_image.SetDirection(meta_data["direction"])

            d[k] = itk_image
        return d
