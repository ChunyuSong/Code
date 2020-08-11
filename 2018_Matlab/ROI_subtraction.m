(* ::Package:: *)

%%% generate ROI for FLAIR hyperintensity without Gd_T1W hyperintensity regions
clear;
dirROI = ['C:\Users\Desktop\GBM\GBM_3\'];

filename = fullfile(dirROI,'Gd_hyper.nii.gz');
mask = load_nii(filename);
data_Gd _hyper = mask.img;

filename = fullfile(dirROI,'FLAIR_hyper.nii.gz');
mask = load_nii(filename);
data_FLAIR = mask.img;

% filename = fullfile(dirROI,'Gd_hypo.nii.gz');
% mask = load_nii(filename);
% data_Gd _hypo = mask.img;

% data_FLAIR _Gd = (data_FLAIR - data_Gd _hyper - data_Gd _hypo) > 0;
data_FLAIR _Gd = (data_FLAIR - data_Gd _hyper) > 0;
data_FLAIR _Gd = uint8(data_FLAIR _Gd);
imagesc(data_FLAIR(:,:,100));
nii = make_nii(data_FLAIR _Gd);
save_nii(nii,'FLAIR_Gd.nii.gz');






