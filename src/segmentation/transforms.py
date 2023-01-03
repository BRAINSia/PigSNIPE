import itk
import numpy as np
import torch
import re


def clip_with_mask(img, mask):
    img_arr = itk.GetArrayFromImage(img)
    mask_arr = itk.GetArrayFromImage(mask)
    new_arr = img_arr * mask_arr
    new_im = itk.GetImageFromArray(new_arr)
    new_im.CopyInformation(img)
    return new_im


class SkullStripd(object):
    def __init__(self, keys):
        self.im_keys = keys[:-1]
        self.lbl_key = keys[-1]

    def __call__(self, data):
        d = dict(data)
        label = d[self.lbl_key]
        for key in self.im_keys:
            image = d[key]
            stripped_image = clip_with_mask(image, label)
            d[key] = stripped_image

        return d


class CropAndStripByResampled(object):
    def __init__(
        self,
        keys,
        inference=False,
        skullstrip=True,
        size=[128, 224, 160],
        spacing=[0.5, 0.5, 0.5],
        adjust_centroid=[0, 0, -8],
    ):
        self.im_keys = keys[:-1]
        self.lbl_key = keys[-1]
        self.inference = inference
        self.strip = skullstrip
        self.size = size
        self.spacing = spacing
        self.adjust_centroid = adjust_centroid

    def __call__(self, data):
        d = dict(data)
        label = d[self.lbl_key]
        label_type = type(label)
        ref_image = self.setup_reference_image(label, pixel_type=itk.F)
        identity_transform = itk.IdentityTransform[itk.D, 3].New()
        for key in self.im_keys:
            image = d[key]
            # resample icv mask for skull stripping
            lbl_interpolator = itk.NearestNeighborInterpolateImageFunction[
                label_type, itk.D
            ].New()
            lbl_resampler = itk.ResampleImageFilter[label_type, label_type].New()
            lbl_resampler.SetInterpolator(lbl_interpolator)
            lbl_resampler.SetTransform(identity_transform)
            lbl_resampler.SetInput(label)
            lbl_resampler.SetReferenceImage(image)
            lbl_resampler.UseReferenceImageOn()
            lbl_resampler.UpdateLargestPossibleRegion()
            resampled_label = lbl_resampler.GetOutput()

            if self.strip:
                image = clip_with_mask(image, resampled_label)

            if not self.inference and self.im_keys.index(key) + 1 != len(self.im_keys):
                image_type = type(image)
                interpolator = itk.LinearInterpolateImageFunction[
                    image_type, itk.D
                ].New()

                resampler = itk.ResampleImageFilter[image_type, image_type].New()
                resampler.SetInterpolator(interpolator)
                resampler.SetTransform(identity_transform)
                resampler.SetInput(image)
                resampler.SetReferenceImage(ref_image)
                resampler.UseReferenceImageOn()
                resampler.UpdateLargestPossibleRegion()
            else:
                image_type = type(image)
                interpolator = itk.NearestNeighborInterpolateImageFunction[
                    image_type, itk.D
                ].New()
                resampler = itk.ResampleImageFilter[image_type, image_type].New()
                resampler.SetInterpolator(interpolator)
                resampler.SetTransform(identity_transform)
                resampler.SetInput(image)
                resampler.SetReferenceImage(ref_image)
                resampler.UseReferenceImageOn()
                resampler.UpdateLargestPossibleRegion()

            # update the dictionary
            d[key] = resampler.GetOutput()
        return d

    def get_image_center(self, image):
        moments_calc = itk.ImageMomentsCalculator[type(image)].New()
        moments_calc.SetImage(image)
        moments_calc.Compute()
        centroid = moments_calc.GetCenterOfGravity()
        centroid = np.add(centroid, np.array(self.adjust_centroid)).tolist()
        return centroid

    def setup_reference_image(self, rough_label, pixel_type):
        center = self.get_image_center(rough_label)
        size = itk.Size[3]()
        size[0] = self.size[0]  # width L-R
        size[1] = self.size[1]  # length Front Back
        size[2] = self.size[2]  # Height top Bottom

        start = itk.Index[3]()
        start[0] = 0
        start[1] = 0
        start[2] = 0

        origin = np.array(center) - (
            np.array([(size[0] / 2 - 1), (size[1] / 2 - 1), (size[2] / 2 - 1)])
            * np.array(self.spacing)
        )
        ImageType = itk.Image[pixel_type, 3]
        fixed_field = ImageType.New()
        fixed_field.SetOrigin(origin.tolist())
        fixed_field.SetSpacing(self.spacing)

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        region.SetIndex(start)

        fixed_field.SetRegions(region)
        fixed_field.Allocate()

        return fixed_field


class ResampleMaskToOgd(object):
    def __init__(self, keys):
        # assert len(keys) == 2, "must pass in a t1w key and label key"
        self.label_key = keys[0]
        self.og_image_key = keys[1]

    def __call__(self, data):
        d = dict(data)
        itk_label = d[self.label_key]
        itk_og_image = d[self.og_image_key]
        image_type = itk.Image[itk.F, 3]
        label_type = itk.Image[itk.UC, 3]
        castImageFilter = itk.CastImageFilter[image_type, label_type].New()
        castImageFilter.SetInput(itk_label)

        nearest_interpolator = itk.NearestNeighborInterpolateImageFunction[
            label_type, itk.D
        ].New()
        identity_transform = itk.IdentityTransform[itk.D, 3].New()

        label_resampler = itk.ResampleImageFilter[label_type, label_type].New()
        label_resampler.SetInterpolator(nearest_interpolator)
        label_resampler.SetTransform(identity_transform)
        label_resampler.SetInput(castImageFilter.GetOutput())
        label_resampler.SetReferenceImage(itk_og_image)
        label_resampler.UseReferenceImageOn()
        label_resampler.UpdateLargestPossibleRegion()

        # update the dictionary
        d[self.label_key] = label_resampler.GetOutput()
        return d


class ResampleRoughRegiond(object):
    def __init__(self, keys, spacing, size, inference=False):
        self.n = len(keys) == 2
        self.inference = inference
        self.spacing = spacing
        self.im_size = size
        if self.inference:  # "must pass in a t1w key and label key"
            self.t1w_key = keys[0]
        else:
            self.t1w_key = keys[0]
            self.label_key = keys[1]

    def __call__(self, data):
        d = dict(data)
        t1w_itk_image = d[self.t1w_key]
        image_type = itk.Image[itk.F, 3]
        linear_interpolator = itk.LinearInterpolateImageFunction[
            image_type, itk.D
        ].New()
        identity_transform = itk.IdentityTransform[itk.D, 3].New()

        center = self.get_image_center(t1w_itk_image)
        ref_image = self.setup_reference_image(
            self.spacing, self.im_size, center, pixel_type=itk.F
        )
        t1w_resampler = itk.ResampleImageFilter[image_type, image_type].New()
        t1w_resampler.SetInterpolator(linear_interpolator)
        t1w_resampler.SetTransform(identity_transform)
        t1w_resampler.SetInput(t1w_itk_image)
        t1w_resampler.SetReferenceImage(ref_image)
        t1w_resampler.UseReferenceImageOn()
        t1w_resampler.UpdateLargestPossibleRegion()
        d[self.t1w_key] = t1w_resampler.GetOutput()

        if not self.inference:
            label_itk_image = d[self.label_key]
            label_type = itk.Image[itk.UC, 3]
            nearest_interpolator = itk.NearestNeighborInterpolateImageFunction[
                label_type, itk.D
            ].New()

            label_resampler = itk.ResampleImageFilter[label_type, label_type].New()
            label_resampler.SetInterpolator(nearest_interpolator)
            label_resampler.SetTransform(identity_transform)
            label_resampler.SetInput(label_itk_image)
            label_resampler.SetReferenceImage(ref_image)
            label_resampler.UseReferenceImageOn()
            label_resampler.UpdateLargestPossibleRegion()

            # update the dictionary
            d[self.label_key] = label_resampler.GetOutput()
        return d

    def get_image_center(self, image):
        size = image.GetLargestPossibleRegion().GetSize()
        center = ((np.array(size) / 2) - 1).astype(np.int32)
        center_loc = image.TransformIndexToPhysicalPoint(center.tolist())
        return center_loc

    def setup_reference_image(self, spacing, im_size, center, pixel_type):
        im_spacing = spacing

        size = itk.Size[3]()
        size[0] = im_size[0]  # size along X
        size[1] = im_size[1]  # size along Y
        size[2] = im_size[2]  # size along Z

        start = itk.Index[3]()
        start[0] = 0  # first index on Xcd
        start[1] = 0  # first index on Y
        start[2] = 0  # first index on Z

        origin = np.array(center) - (
            np.array([(size[0] / 2 - 1), (size[1] / 2 - 1), (size[2] / 2 - 1)])
            * np.array(im_spacing)
        )
        ImageType = itk.Image[pixel_type, 3]
        fixed_field = ImageType.New()
        fixed_field.SetOrigin(origin.tolist())
        fixed_field.SetSpacing(im_spacing)

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        region.SetIndex(start)

        fixed_field.SetRegions(region)
        fixed_field.Allocate()

        return fixed_field


class LoadITKImaged(object):
    def __init__(self, keys, pixel_types):
        self.keys = keys
        self.pixel_types = pixel_types
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        for key, type in zip(self.keys, self.pixel_types):
            # save off the file name
            if f"{key}_meta_dict" not in d.keys():
                d[f"{key}_meta_dict"] = {"filename": d[key]}
            else:
                d[f"{key}_meta_dict"]["filename"] = d[key]

            d[key] = itk.imread(d[key], type)
        d = self.meta_updater(d)

        return d


def get_direction_cos_from_image(image):
    dims = len(image.GetOrigin())
    arr = np.array([[0.0] * dims] * dims)
    mat = image.GetDirection().GetVnlMatrix()
    for i in range(dims):
        for j in range(dims):
            arr[i][j] = mat.get(i, j)
    return arr


class UpdateMetaDatad(object):
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


# conversion functions
class ITKImageToNumpyd(object):
    def __init__(self, keys):
        self.keys = keys
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        d = self.meta_updater(d)
        for k in self.keys:
            d[k] = itk.array_from_image(d[k])

        return d


class ToITKImaged(object):
    def __init__(self, keys):
        self.keys = keys
        pass

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if torch.is_tensor(d[k]):
                d[k] = d[k].numpy().astype(np.float32)
            if len(d[k].shape) == 5:
                d[k] = d[k].squeeze(axis=0).squeeze(axis=0)
            elif len(d[k].shape) == 4:
                d[k] = d[k].squeeze(axis=0)

            meta_data = d[f"{k}_meta_dict"]
            itk_image = itk.image_from_array(d[k])
            itk_image.SetOrigin(meta_data["origin"])
            itk_image.SetSpacing(meta_data["spacing"])
            itk_image.SetDirection(meta_data["direction"])

            d[k] = itk_image
        return d


def sub_from_path(file_path):
    sub = re.findall("sub-\\w+\\d+", file_path)[0]
    return sub


class SaveITKImaged(object):
    def __init__(
        self,
        keys,
        output_filenames: list,
    ):
        self.keys = keys
        self.out_filenames = output_filenames

    def __call__(self, data):
        d = dict(data)
        for key, out_fname in zip(self.keys, self.out_filenames):
            print("Writing to:", out_fname)
            itk.imwrite(d[key], out_fname)

        return d


class AddChanneld(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            im = d[k]
            im = np.expand_dims(im, axis=0)
            d[k] = im

        return d


unsqueze_lambda = lambda x: x.squeeze(dim=0)
shape_lambda = lambda x: x.shape


class CombineImagesd(object):
    def __init__(self, keys):
        assert len(keys) == 2
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        im1 = d[self.keys[0]]
        im2 = d[self.keys[1]]
        assert im1.shape == im2.shape
        shape = im1.shape
        new_image = np.empty([2, shape[0], shape[1], shape[2]])
        new_image[0] = im1
        new_image[1] = im2

        d["image"] = new_image.astype(np.float32)
        return d


class CropByResampled(object):
    def __init__(self, keys):
        # assert len(keys) == 2, "must pass in a t1w key and label key"
        self.t1w_key = keys[0]
        self.label_key = keys[1]

    def __call__(self, data):
        d = dict(data)
        t1w_itk_image = d[self.t1w_key]
        label_itk_image = d[self.label_key]

        ref_image = self.setup_reference_image(label_itk_image, pixel_type=itk.F)
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

        label_resampler = itk.ResampleImageFilter[label_type, label_type].New()
        label_resampler.SetInterpolator(nearest_interpolator)
        label_resampler.SetTransform(identity_transform)
        label_resampler.SetInput(label_itk_image)
        label_resampler.SetReferenceImage(ref_image)
        label_resampler.UseReferenceImageOn()
        label_resampler.UpdateLargestPossibleRegion()

        # update the dictionary
        d[self.t1w_key] = t1w_resampler.GetOutput()
        d[self.label_key] = label_resampler.GetOutput()
        return d

    def get_image_center(self, image):
        moments_calc = itk.ImageMomentsCalculator[type(image)].New()
        moments_calc.SetImage(image)
        moments_calc.Compute()
        centroid = moments_calc.GetCenterOfGravity()
        a = np.random.uniform(low=-5, high=5, size=(3,))
        centroid = np.add(centroid, a).tolist()
        return centroid

    def setup_reference_image(self, rough_label, pixel_type):
        center = self.get_image_center(rough_label)
        im_spacing = [0.5, 0.5, 0.5]
        size = itk.Size[3]()
        size[0] = 224  # size along X
        size[1] = 224  # size along Y
        size[2] = 160  # size along Z

        start = itk.Index[3]()
        start[0] = 0  # first index on Xcd
        start[1] = 0  # first index on Y
        start[2] = 0  # first index on Z

        origin = np.array(center) - (
            np.array([(size[0] / 2 - 1), (size[1] / 2 - 1), (size[2] / 2 - 1)])
            * np.array(im_spacing)
        )
        ImageType = itk.Image[pixel_type, 3]
        fixed_field = ImageType.New()
        fixed_field.SetOrigin(origin.tolist())
        fixed_field.SetSpacing(im_spacing)

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        region.SetIndex(start)

        fixed_field.SetRegions(region)
        fixed_field.Allocate()

        return fixed_field
