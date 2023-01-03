![](/pigsnipe_logo.png)

<div align="center">
  <p align="center">Scalable Neuroimaging Processing Engine for Minipig MRI</p>
</div>

## Introduction

PigSNIPE is a software package for analysis of Minipig brain Magnetic Resonance Images (MRI). It is developed by the [SINAPSE lab](https://medicine.uiowa.edu/psychiatry/sinapse) at 
[the University of Iowa, Electrical and Computer Engineering department](https://ece.engineering.uiowa.edu/).

PigSNIPE provides a fully automatic pipeline that allows for image registration, AC-PC alignment, brain mask segmentation, skull stripping, tissue segmentation, caudate-putamen brain segmentation, and landmark detection.

For detailed information we refer you to the [PigSNIPE paper]().

## Setup

1. Clone this git repository.

    `$ git clone ...`

2. Download the [zip file](https://iowa.sharepoint.com/:u:/r/sites/SINAPSELAB/Shared%20Documents/PigSNIPE/DL_MODEL_PARAMS.zip?csf=1&web=1&e=8y0BX4) containing model weights.

3. Unzip the file and place the directory in the cloned repo.

    `$ unzip <path_to_zip_file> -d <path_to_repo> `

4. Build docker image.

    `$ docker build -it pigsnipe .`

5. Run docker image.

    `$ docker run pigsnipe`

## Authors

Michal Brzus

Hans Johnson, Ph.D
