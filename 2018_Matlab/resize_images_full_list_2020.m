%% flip the DBSI.nii maps by yzz on 04/13/2016

% generate DWI_1500 image
nii = load_nii('dti_adc_map.nii');
adc = nii.img;
b0 = load_nii('b0_map.nii');
b0 = b0.img;
c = b0.*exp(-1.5*adc);
c = flipdim(c, 2) ;  % flip iamges vertically
c = imresize(c, 2); % interpolate iamge with bicubic method
nii = make_nii(c);
save_nii(nii, 'dwi_1500_map.nii');

% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'b0_map.nii');

% DTI_ADC
nii = load_nii('dti_adc_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_adc_map.nii');

% DTI_Axial
nii = load_nii('dti_axial_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_axial_map.nii');

% DTI_Radial
nii = load_nii('dti_radial_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_radial_map.nii');

% DTI_FA
nii = load_nii('dti_fa_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_fa_map.nii');

% Fiber_ratio
nii = load_nii('fiber_ratio_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber_ratio_map.nii');

% Fiber1_axial
nii = load_nii('fiber1_axial_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_axial_map.nii');

% Fiber1_radial
nii = load_nii('fiber1_radial_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_radial_map.nii');

% Fiber1_FA
nii = load_nii('fiber1_fa_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_fa_map.nii');

% Fiber1_Fiber_Ratio
nii = load_nii('fiber1_fiber_ratio_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_fiber_ratio_map.nii');

% Fiber2_axial
nii = load_nii('fiber2_axial_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_axial_map.nii');

% Fiber2_radial
nii = load_nii('fiber2_radial_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_radial_map.nii');

% Fiber1_FA
nii = load_nii('fiber2_fa_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_fa_map.nii');

% Fiber1_Fiber_Ratio
nii = load_nii('fiber2_fiber_ratio_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_fiber_ratio_map.nii');

% restricted_ratio_1
nii = load_nii('restricted_ratio_1_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_ratio_1_map.nii');

% restricted_adc_1
nii = load_nii('restricted_adc_1_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_adc_1_map.nii');

% restricted_ratio_2
nii = load_nii('restricted_ratio_2_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_ratio_2_map.nii');

% restricted_adc_2
nii = load_nii('restricted_adc_2_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_adc_2_map.nii');

% hindered_ratio
nii = load_nii('hindered_ratio_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'hindered_ratio_map.nii');

% hindered_adc
nii = load_nii('hindered_adc_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'hindered_adc_map.nii');

% water_ratio
nii = load_nii('water_ratio_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'water_ratio_map.nii');

% water_adc
nii = load_nii('water_adc_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'water_adc_map.nii');

% iso_adc
nii = load_nii('iso_adc_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'iso_adc_map.nii');

disp('interpolation and flip: complete!!!!');


















