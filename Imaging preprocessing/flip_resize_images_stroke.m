(* ::Package:: *)



% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
b0 = imresize(img_ 1,2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii,'b0.nii');

% DTI_ADC
nii=load_nii('dti_adc _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_ADC=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_ADC);
save_nii(nii,'DTI_ADC.nii');

% DTI_FA
nii=load_nii('dti_fa _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_FA=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_FA);
save_nii(nii,'DTI_FA.nii');

% fiber_ratio
nii=load_nii('fiber_ratio _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
fiber=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(fiber);
save_nii(nii,'fiber.nii');

% restricted_ratio _ 1;
nii=load_nii('restricted_ratio _ 1_map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
restricted_ 1=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_ 1);
save_nii(nii,'restricted_ 1.nii');

% restricted_ratio _ 2;
nii=load_nii('restricted_ratio _ 2_map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
restricted_ 2=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_ 2);
save_nii(nii,'restricted_ 2.nii');

% hindered_ratio
nii=load_nii('hindered_ratio _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
hindered=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(hindered);
save_nii(nii,'hindered.nii');

% water_ratio
nii=load_nii('water_ratio _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
water=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(water);
save_nii(nii,'water.nii');

% generate DWI_ 1500 image
nii = load_nii('dti_adc _map.nii');
adc=nii.img;
b0=load_nii('b0_map.nii');
b0=b0.img;
img=b0.*exp(-1.5*adc);
img_ 1 = fliplr(img); % flip iamges vertically
dwi_ 1500=imresize(img_ 1,2); % interpolate iamgew with bi cubic method
nii = make_nii(dwi_ 1500);
save_nii(nii,'dwi_ 1500.nii');

% dti_axial
nii=load_nii('dti_axial _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_Axial = imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_Axial);
save_nii(nii,'DTI_Axial.nii');

% dti_radial
nii=load_nii('dti_radial _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_Radial=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_Radial);
save_nii(nii,'DTI_Radial.nii');

% dbsi_axial
nii=load_nii('fiber1_axial _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DBSI_Axial = imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DBSI_Axial);
save_nii(nii,'DBSI_Axial.nii');

% dbsi_radial
nii=load_nii('fiber1_radial _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DBSI_Radial = imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DBSI_Radial);
save_nii(nii,'DBSI_Radial.nii');

% dti_rgba _itk
nii=load_nii('dti_rgba _map _itk.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_RGBA _ITK = imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_RGBA _ITK);
save_nii(nii,'DTI_RGBA _ITK.nii');

% dbsi_rgba _itk
nii=load_nii('fiber1_rgba _map _itk.nii');
img=nii.img;
img_ 1 = fliplr(img) % flip iamges vertically;
DBSI_RGBA _ITK = imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DBSI_RGBA _ITK);
save_nii(nii,'DBSI_RGBA _ITK.nii');















