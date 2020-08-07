%% flip the DBSI.nii maps by yzz on 04/13/2016

% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img_1 = fliplr(img); % flip iamges vertically
b0 = imresize(img_1,2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii,'b0.nii');

% DTI_ADC
nii=load_nii('dti_adc_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_ADC=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_ADC);
save_nii(nii,'DTI_ADC.nii');

% DTI_FA
nii=load_nii('dti_fa_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_FA=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_FA);
save_nii(nii,'DTI_FA.nii');

% fiber_ratio
nii=load_nii('fiber_ratio_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
fiber=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(fiber);
save_nii(nii,'fiber.nii');

% restricted_ratio_1;
nii=load_nii('restricted_ratio_1_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
restricted_1=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_1);
save_nii(nii,'restricted_1.nii');

% restricted_ratio_2;
nii=load_nii('restricted_ratio_2_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
restricted_2=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_2);
save_nii(nii,'restricted_2.nii');

% hindered_ratio
nii=load_nii('hindered_ratio_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
hindered=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(hindered);
save_nii(nii,'hindered.nii');

% water_ratio
nii=load_nii('water_ratio_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
water=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(water);
save_nii(nii,'water.nii');

% generate DWI_1500 image
nii = load_nii('dti_adc_map.nii');
adc=nii.img;
b0=load_nii('b0_map.nii');
b0=b0.img;
img=b0.*exp(-1.5*adc);
img_1 = fliplr(img); % flip iamges vertically
dwi_1500=imresize(img_1,2); % interpolate iamgew with bi cubic method
nii = make_nii(dwi_1500);
save_nii(nii,'dwi_1500.nii');

% dti_axial
nii=load_nii('dti_axial_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_Axial = imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_Axial);
save_nii(nii,'DTI_Axial.nii');

% dti_radial
nii=load_nii('dti_radial_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_Radial=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_Radial);
save_nii(nii,'DTI_Radial.nii');

% dbsi_axial
nii=load_nii('fiber1_axial_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DBSI_Axial = imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DBSI_Axial);
save_nii(nii,'DBSI_Axial.nii');

% dbsi_radial
nii=load_nii('fiber1_radial_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DBSI_Radial = imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DBSI_Radial);
save_nii(nii,'DBSI_Radial.nii');

% dti_rgba_itk
nii=load_nii('dti_rgba_map_itk.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_RGBA_ITK = imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_RGBA_ITK);
save_nii(nii,'DTI_RGBA_ITK.nii');

% dbsi_rgba_itk
nii=load_nii('fiber1_rgba_map_itk.nii');
img=nii.img;
img_1 = fliplr(img) % flip iamges vertically;
DBSI_RGBA_ITK = imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DBSI_RGBA_ITK);
save_nii(nii,'DBSI_RGBA_ITK.nii');















