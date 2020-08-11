(* ::Package:: *)




% DTI_FA
nii = load_untouch _nii('dti_fa _map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'dti_fa.nii');

% Fiber_ratio
nii = load_untouch _nii('fiber_ratio _map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'fiber_fraction.nii');

% highly_restricted _fraction
nii = load_untouch _nii('restricted_ratio _ 1_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'highly_restricted _fraction.nii');

% restricted_fraction
nii = load_untouch _nii('restricted_ratio _ 2_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'restricted_fraction.nii');

% hindered_ratio
nii = load_untouch _nii('hindered_ratio _map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'hindered_fraction.nii');

% water_ratio
nii = load_untouch _nii('water_ratio _map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'water_fraction.nii');

disp('interpolation: complete!!!!');


















