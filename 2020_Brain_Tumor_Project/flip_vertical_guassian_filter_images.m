(* ::Package:: *)



% b0
nii = load_untouch _nii('b0_map.nii');
img = nii.img;
img = flipdim(img, 2);
nii = make_nii(img);
save_nii(nii, 'b0.nii');

% DTI_ADC
nii = load_untouch _nii('dti_adc _map.nii');
img = nii.img;
img = flipdim(img, 2);
nii = make_nii(img);
save_nii(nii, 'dti_adc.nii');

% DTI_FA
nii = load_untouch _nii('dti_fa _map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'dti_fa.nii');


% Fiber_ratio
nii = load_untouch _nii('fiber_fraction _map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'fiber_fraction.nii');

% restricted_ratio _ 1
nii = load_untouch _nii('highly_restricted _fraction _map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'highly_restricted _fraction.nii');

% restricted_ratio _ 2
nii = load_untouch _nii('restricted_fraction _map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'restricted_fraction.nii');

% hindered_ratio
nii = load_untouch _nii('hindered_fraction _map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'hindered_fraction.nii');

% water_ratio
nii = load_untouch _nii('water_fraction _map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'water_fraction.nii');

disp('interpolation and flip: complete!!!!')


















