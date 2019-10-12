#!/bin/bash

OLDIFS=$IFS
IFS=","

while read id notid
do

    echo "$id"
    aws s3 sync s3://hcp-openaccess-temp/HCP_1200/$id/MNINonLinear/Results/rfMRI_REST1_LR/ --profile HCP ~/data/HCPHundredUnrelated/$id/ --exclude "*" --include "rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii" --dryrun
    aws s3 sync s3://hcp-openaccess-temp/HCP_1200/$id/MNINonLinear/Results/rfMRI_REST1_RL/ --profile HCP ~/data/HCPHundredUnrelated/$id/ --exclude "*" --include "rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii" --dryrun
    aws s3 sync s3://hcp-openaccess-temp/HCP_1200/$id/MNINonLinear/Results/rfMRI_REST2_LR/ --profile HCP ~/data/HCPHundredUnrelated/$id/ --exclude "*" --include "rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii" --dryrun
    aws s3 sync s3://hcp-openaccess-temp/HCP_1200/$id/MNINonLinear/Results/rfMRI_REST2_RL/ --profile HCP ~/data/HCPHundredUnrelated/$id/ --exclude "*" --include "rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii" --dryrun

done < $1

IFS=$OLDIFS
