![](/pigsnipe_logo.png)

<div align="center">
  <p align="center">Scalable Neuroimaging Processing Engine for Minipig MRI</p>
</div>

## Introduction

PigSNIPE is a software package for analysis of Minipig brain Magnetic Resonance Images (MRI). It is developed by the [SINAPSE lab](https://medicine.uiowa.edu/psychiatry/sinapse) at 
[the University of Iowa, Electrical and Computer Engineering department](https://ece.engineering.uiowa.edu/).

PigSNIPE provides a fully automatic pipeline that allows for image registration, AC-PC alignment, brain mask segmentation, skull stripping, tissue segmentation, caudate-putamen brain segmentation, and landmark detection.

For detailed information we refer you to the [PigSNIPE paper](https://www.preprints.org/manuscript/202301.0313/v1).

## Setup

1. Clone this git repository.

    `$ git clone https://github.com/BRAINSia/PigSNIPE.git`

2. Download the [zip file](https://iowa.sharepoint.com/:u:/r/sites/SINAPSELAB/Shared%20Documents/PigSNIPE/DL_MODEL_PARAMS.zip?csf=1&web=1&e=8y0BX4) containing model weights. 
  
    Note: The Link Will be available shortly. If you wish to use the tool sooner email hans-johnson@uiowa.edu or michal-brzus@uiowa.edu.

3. Unzip the DL_MODEL_PARAMS directory in the cloned repo.

    `$ unzip <path_to_zip_file> -d <path_to_repo> `

4. Go to the repo directory and build a docker image.

    `$ docker build --tag pigsnipe:v0.9 --label pigsnipe --file $(pwd)/Dockerfile $(pwd)`

5. Run docker image.

    `$ docker run pigsnipe:v0,9`
   
   You should be seeing the script help message.
   
   Example command for using the script would be:
   
    `$ docker run --mount type=bind,source=<data_dir>,target=/tmp/mydata pigsnipe:v0.9 -t1 /tmp/mydata/sub-001_ses-001_T1w.nii.gz -t2 /tmp/mydata/sub-001_ses-002_T2w.nii.gz -o /tmp/mydata/results`
    
    Note: The results would be placed in `<data_dir>/results`
    
    Note: The tool expect the data to be compatible with [BIDS format](https://bids.neuroimaging.io/). 

## Authors

[Michal Brzus](https://github.com/mbrzus) - Ph.D. student at the University of Iowa

[Hans Johnson, Ph.D](https://github.com/hjmjohnson) - [Professor](https://engineering.uiowa.edu/people/hans-johnson) at the University of Iowa, Electrical and Computer Engineering department.

for contact, email: hans-johnson@uiowa.edu
