%% flip the DBSI.nii maps by yzz on 04/13/2016


% DTI_FA
nii = load_untouch_nii('dti_fa_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'dti_fa.nii');

% Fiber_ratio
nii = load_untouch_nii('fiber_ratio_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'fiber_fraction.nii');

% highly_restricted_fraction
nii = load_untouch_nii('restricted_ratio_1_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'highly_restricted_fraction.nii');

% restricted_fraction
nii = load_untouch_nii('restricted_ratio_2_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'restricted_fraction.nii');

% hindered_ratio
nii = load_untouch_nii('hindered_ratio_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'hindered_fraction.nii');

% water_ratio
nii = load_untouch_nii('water_ratio_map.nii');
img = nii.img;
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'water_fraction.nii');

disp('interpolation: complete!!!!');


















