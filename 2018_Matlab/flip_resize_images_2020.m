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
save_nii(nii, 'dwi_1500.nii');

% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img = flipdim(img, 2) ; 
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'b0.nii');

% DTI_ADC
nii = load_nii('dti_adc_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_adc.nii');

% DTI_FA
nii = load_nii('dti_fa_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
nii = make_nii(img);
save_nii(nii, 'dti_fa.nii');


% Fiber_ratio
nii = load_nii('fiber_ratio_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'fiber_ratio.nii');

% restricted_ratio_1
nii = load_nii('restricted_ratio_1_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'restricted_ratio_1.nii');

% restricted_ratio_2
nii = load_nii('restricted_ratio_2_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'restricted_ratio_2.nii');

% hindered_ratio
nii = load_nii('hindered_ratio_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'hindered_ratio.nii');

% water_ratio
nii = load_nii('water_ratio_map.nii');
img = nii.img;
img = flipdim(img, 2);
img = imresize(img, 2); 
img = imgaussfilt(img, 1);
nii = make_nii(img);
save_nii(nii, 'water_ratio.nii');

disp('interpolation and flip: complete!!!!')


















