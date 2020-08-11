(* ::Package:: *)


% generate DWI_ 1500 image
nii = load_nii('dti_adc _map.nii');
adc = nii.img;
b0 = load_nii('b0_map.nii');
b0 = b0.img;
c = b0.*exp(-1.5*adc);
c = flipdim(c, 1) ;
c = flipdim(c, 2) ;  % flip iamges vertically
nii = make_nii(c);
save_nii(nii, 'dwi_ 1500.nii');

% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
nii = make_nii(img);
save_nii(nii, 'b0.nii');

% DTI_ADC
nii = load_nii('dti_adc _map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
nii = make_nii(img);
save_nii(nii, 'dti_adc.nii');

% DTI_FA
nii = load_nii('dti_fa _map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'dti_fa.nii');


% Fiber_ratio
nii = load_nii('fiber_ratio _map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'fiber_ratio.nii');

% % restricted_ratio _ 1
% nii = load_nii('restricted_ratio _ 1_map.nii');
% img = nii.img;
% img = flipdim(img, 1);
% img = flipdim(img, 2);
% img = imgaussfilt(img, 1);
% nii = make_nii(img);
% save_nii(nii, 'restricted_ratio _ 1.nii');

% % restricted_ratio _ 2
% nii = load_nii('restricted_ratio _ 2_map.nii');
% img = nii.img;
% img = flipdim(img, 1);
% img = flipdim(img, 2);
% img = imgaussfilt(img, 1);
% nii = make_nii(img);
% save_nii(nii, 'restricted_ratio _ 2.nii');

nii = load_nii('restricted_ratio _map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'restricted_ratio.nii');

% hindered_ratio
nii = load_nii('hindered_ratio _map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'hindered_ratio.nii');

% water_ratio
nii = load_nii('water_ratio _map.nii');
img = nii.img;
img = flipdim(img, 1);
img = flipdim(img, 2);
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'water_ratio.nii');

disp('interpolation and flip: complete!!!!')


















