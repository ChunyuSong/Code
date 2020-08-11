(* ::Package:: *)


% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img_ 1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 1;
b0 = imresize(img_ 1,2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii,'b0.nii');

% DTI_ADC
nii=load_nii('dti_adc _map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 1;
DTI_ADC=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_ADC);
save_nii(nii,'DTI_ADC.nii');

% DTI_FA
nii=load_nii('dti_fa _map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
DTI_FA=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_FA);
save_nii(nii,'DTI_FA.nii');

% fiber_ratio
nii=load_nii('fiber_ratio _map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
fiber=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(fiber);
save_nii(nii,'fiber.nii');


% restricted_ 1_ratio;
nii=load_nii('restricted_ratio _ 1_map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
restricted_ 1=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_ 1);
save_nii(nii,'restricted_ 1.nii');

% restricted_ 2_ratio;
nii=load_nii('restricted_ratio _ 2_map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
restricted_ 2=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_ 2);
save_nii(nii,'restricted_ 2.nii');

% hindered_ratio
nii=load_nii('hindered_ratio _map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
hindered=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(hindered);
save_nii(nii,'hindered.nii');

% water_ratio
nii=load_nii('water_ratio _map.nii');
img=nii.img;
img_ 1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
water=imresize(img_ 1,2); % interpolate iamgew with bicubic method
nii=make_nii(water);
save_nii(nii,'water.nii');



disp('Generate DBSI Results: Completed!');
















