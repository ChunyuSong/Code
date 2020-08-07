%% flip the DBSI.nii maps by yzz on 04/13/2016

% b0
nii = load_untouch_nii('b0_map.nii');
img = nii.img;
img = flipdim(img, 2);
nii = make_nii(img);
save_nii(nii, 'b0.nii');

% DTI_ADC
nii = load_untouch_nii('dti_adc_map.nii');
img = nii.img;
img = flipdim(img, 2);
nii = make_nii(img);
save_nii(nii, 'dti_adc.nii');

% DTI_FA
nii = load_untouch_nii('dti_fa_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'dti_fa.nii');


% Fiber_ratio
nii = load_untouch_nii('fiber_fraction_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'fiber_fraction.nii');

% restricted_ratio_1
nii = load_untouch_nii('highly_restricted_fraction_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'highly_restricted_fraction.nii');

% restricted_ratio_2
nii = load_untouch_nii('restricted_fraction_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'restricted_fraction.nii');

% hindered_ratio
nii = load_untouch_nii('hindered_fraction_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'hindered_fraction.nii');

% water_ratio
nii = load_untouch_nii('water_fraction_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imgaussfilt(img, 0.8);
nii = make_nii(img);
save_nii(nii, 'water_fraction.nii');

disp('interpolation and flip: complete!!!!')


















