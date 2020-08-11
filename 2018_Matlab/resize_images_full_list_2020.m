(* ::Package:: *)

% generate DWI_ 1500 image
nii = load_nii('dti_adc _map.nii');
adc = nii.img;
b0 = load_nii('b0_map.nii');
b0 = b0.img;
c = b0.*exp(-1.5*adc);
c = flipdim(c, 2) ;  % flip iamges vertically
c = imresize(c, 2); % interpolate iamge with bicubic method
nii = make_nii(c);
save_nii(nii, 'dwi_ 1500_map.nii');

% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'b0_map.nii');

% DTI_ADC
nii = load_nii('dti_adc _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_adc _map.nii');

% DTI_Axial
nii = load_nii('dti_axial _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_axial _map.nii');

% DTI_Radial
nii = load_nii('dti_radial _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_radial _map.nii');

% DTI_FA
nii = load_nii('dti_fa _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_fa _map.nii');

% Fiber_ratio
nii = load_nii('fiber_ratio _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber_ratio _map.nii');

% Fiber1_axial
nii = load_nii('fiber1_axial _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_axial _map.nii');

% Fiber1_radial
nii = load_nii('fiber1_radial _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_radial _map.nii');

% Fiber1_FA
nii = load_nii('fiber1_fa _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_fa _map.nii');

% Fiber1_Fiber _Ratio
nii = load_nii('fiber1_fiber _ratio _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber1_fiber _ratio _map.nii');

% Fiber2_axial
nii = load_nii('fiber2_axial _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_axial _map.nii');

% Fiber2_radial
nii = load_nii('fiber2_radial _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_radial _map.nii');

% Fiber1_FA
nii = load_nii('fiber2_fa _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_fa _map.nii');

% Fiber1_Fiber _Ratio
nii = load_nii('fiber2_fiber _ratio _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'fiber2_fiber _ratio _map.nii');

% restricted_ratio _ 1
nii = load_nii('restricted_ratio _ 1_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_ratio _ 1_map.nii');

% restricted_adc _ 1
nii = load_nii('restricted_adc _ 1_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_adc _ 1_map.nii');

% restricted_ratio _ 2
nii = load_nii('restricted_ratio _ 2_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_ratio _ 2_map.nii');

% restricted_adc _ 2
nii = load_nii('restricted_adc _ 2_map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'restricted_adc _ 2_map.nii');

% hindered_ratio
nii = load_nii('hindered_ratio _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'hindered_ratio _map.nii');

% hindered_adc
nii = load_nii('hindered_adc _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'hindered_adc _map.nii');

% water_ratio
nii = load_nii('water_ratio _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'water_ratio _map.nii');

% water_adc
nii = load_nii('water_adc _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'water_adc _map.nii');

% iso_adc
nii = load_nii('iso_adc _map.nii');
img = nii.img;
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'iso_adc _map.nii');

disp('interpolation and flip: complete!!!!');


















