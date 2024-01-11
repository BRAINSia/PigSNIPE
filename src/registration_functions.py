import itk
from subprocess import call
import sys


def combine_transforms(transforms_arr: list):
    composite = itk.CompositeTransform[itk.D, 3].New()
    for i in transforms_arr:
        t = itk.transformread(i)[0]
        composite.AddTransform(t)

    return composite


def resample_from_comp(image, transform, out_image, refference_image, pixel_type):
    im = itk.imread(image, pixel_type)
    image_type = itk.Image[pixel_type, 3]
    if pixel_type == itk.F:
        interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    else:
        interpolator = itk.NearestNeighborInterpolateImageFunction[
            image_type, itk.D
        ].New()
    ref_im = itk.imread(refference_image, itk.F)
    resampler = itk.ResampleImageFilter[image_type, image_type].New()
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(transform)
    resampler.SetInput(im)
    resampler.SetReferenceImage(ref_im)
    resampler.UseReferenceImageOn()
    resampler.UpdateLargestPossibleRegion()
    print(f"\nwriting to {out_image}\n")
    itk.imwrite(resampler.GetOutput(), out_image)


def call_BRAINSFIT(
    fixed_image, moving_image, transform, fixed_mask=None, moving_mask=None
):
    command = (
        f"BRAINSToolsBinaries/BRAINSFit --fixedVolume {fixed_image} --fixedBinaryVolume {fixed_mask} "
        f"--movingVolume {moving_image} --movingBinaryVolume {moving_mask} --samplingPercentage 0.2 "
        f"--linearTransform {transform} --initializeTransformMode useCenterOfROIAlign --useRigid --useAffine "
        f"--maskProcessingMode ROI  --medianFilterSize 1,1,1 "
        f"--interpolationMode Linear "
    )

    return command


def call_BRAINSFITRigid(
    fixed_image, moving_image, transform, fixed_mask=None, moving_mask=None
):
    command = (
        f"BRAINSToolsBinaries/BRAINSFit --fixedVolume {fixed_image} --fixedBinaryVolume {fixed_mask} "
        f"--movingVolume {moving_image} --movingBinaryVolume {moving_mask} --samplingPercentage 0.2 "
        f"--linearTransform {transform} --initializeTransformMode useCenterOfROIAlign --useRigid "
        f"--maskProcessingMode ROI  --medianFilterSize 1,1,1 "
        f"--interpolationMode Linear "
    )

    return command


def call_BRAINSResample(
    inputim: str, outputim: str, transform: str, ref_vol: str = None
) -> str:
    command = f"BRAINSToolsBinaries/BRAINSResample --inputVolume {inputim}  --outputVolume {outputim} --pixelType float --warpTransform {transform} "
    if ref_vol is not None:
        command = f"{command} --referenceVolume {ref_vol}"
    return command


def call_BRAINSResampleInPlace(
    inputim: str, outputim: str, transform: str, ref_vol: str = None
) -> str:
    command = (
        f"BRAINSToolsBinaries/BRAINSResample --inputVolume {inputim}  --outputVolume {outputim} --pixelType float "
        f"--warpTransform {transform} --interpolationMode ResampleInPlace"
    )
    if ref_vol is not None:
        command = f"{command} --referenceVolume {ref_vol}"
    return command


def call_BRAINSResampleMask(
    inputim: str, outputim: str, transform: str, ref_vol: str = None
) -> str:
    command = (
        f"BRAINSToolsBinaries/BRAINSResample --inputVolume {inputim}  --outputVolume {outputim} --pixelType float "
        f"--warpTransform {transform} --interpolationMode NearestNeighbor "
    )
    if ref_vol is not None:
        command = f"{command} --referenceVolume {ref_vol}"
    return command


def skull_strip(im_path, mask_path, out_im_path):
    img = itk.imread(im_path)
    mask = itk.imread(mask_path, itk.UC)
    img_arr = itk.GetArrayFromImage(img)
    mask_arr = itk.GetArrayFromImage(mask)
    new_arr = img_arr * mask_arr
    new_im = itk.GetImageFromArray(new_arr)
    new_im.CopyInformation(img)
    itk.imwrite(new_im, out_im_path)
