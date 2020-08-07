%%% generate ROI for FLAIR hyperintensity without Gd_T1W hyperintensity regions
clear;
dirROI = ['C:\Users\zye01\Desktop\GBM\GBM_3\'];

filename = fullfile(dirROI,'Gd_hyper.nii.gz');
mask = load_nii(filename);
data_Gd_hyper = mask.img;

filename = fullfile(dirROI,'FLAIR_hyper.nii.gz');
mask = load_nii(filename);
data_FLAIR = mask.img;

% filename = fullfile(dirROI,'Gd_hypo.nii.gz');
% mask = load_nii(filename);
% data_Gd_hypo = mask.img;

%data_FLAIR_Gd = (data_FLAIR - data_Gd_hyper - data_Gd_hypo) > 0;
data_FLAIR_Gd = (data_FLAIR - data_Gd_hyper) > 0;
data_FLAIR_Gd = uint8(data_FLAIR_Gd);
imagesc(data_FLAIR(:,:,100));
nii = make_nii(data_FLAIR_Gd);
save_nii(nii,'FLAIR_Gd.nii.gz');






