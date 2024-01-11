![](/pigsnipe_logo.png)

<div align="center">
  <p align="center">Scalable Neuroimaging Processing Engine for Minipig MRI</p>
</div>

## Introduction

PigSNIPE is a software package for analysis of Minipig brain Magnetic Resonance Images (MRI). It is developed by the [SINAPSE lab](https://medicine.uiowa.edu/psychiatry/sinapse) at 
[the University of Iowa, Electrical and Computer Engineering department](https://ece.engineering.uiowa.edu/).

PigSNIPE provides a fully automatic pipeline that allows for image registration, AC-PC alignment, brain mask segmentation, skull stripping, tissue segmentation, caudate-putamen brain segmentation, and landmark detection.

For detailed information we refer you to the [PigSNIPE paper](https://www.mdpi.com/1999-4893/16/2/116).

## Setup

1. Clone this git repository.

    `$ git clone https://github.com/BRAINSia/PigSNIPE.git`
    
2. Build BRAINSTools

     Refer to the [BRAINSTools](https://github.com/BRAINSia/BRAINSTools) GitHub repository for specific build instructions.

     NOTE: to optimize the build for your machine hardware, use 'NATIVE' mode in ccmake setup.

3. Setting Up Binaries and Libraries.

     To set up binaries and libraries needed to utilize this repository, run the following script inside the PigSNIPE repository.
   
     `$ python3 setup.py -b <path_to_BRAINSTools_build_dir>`

4. Download the [zip file](https://iowa.sharepoint.com/:u:/r/sites/SINAPSELAB/Shared%20Documents/PigSNIPE/DL_MODEL_PARAMS.zip?csf=1&web=1&e=8y0BX4) containing model weights. 
  
    Note: The Link Will be available shortly. If you wish to use the tool sooner email hans-johnson@uiowa.edu or michal-brzus@uiowa.edu.

5. Unzip the DL_MODEL_PARAMS directory in the cloned repo.

    `$ unzip <path_to_zip_file> -d <path_to_repo> `

6. Create a virtual environment and install required packages.
   
    `$ python3 -m venv <path_to_virtual_env>`
        
    `$ source <path_to_virtual_env>/bin/activate`
   
    `$ pip install -r <path_to_REQUIREMENTS.txt>`

7. Run PigSNIPE pipeline

    `$ python3 pigsnipe`

    You should the help message for the script.
   
    The example command to run the pipeline is:
   
    `$ python3 pigsnipe -t1 <path_to_T1w> -t2 <path_to_T2w> -o <path_to_result_directory> --keep_temp_files`


## Authors

[Michal Brzus](https://github.com/mbrzus) - Ph.D. student at the University of Iowa

[Hans Johnson, Ph.D](https://github.com/hjmjohnson) - [Professor](https://engineering.uiowa.edu/people/hans-johnson) at the University of Iowa, Electrical and Computer Engineering department.

for contact, email: hans-johnson@uiowa.edu
