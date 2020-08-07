%% flip the DBSI.nii maps by yzz on 04/13/2016
%% edited on 05/05/2016 interpolate maps by twice

% b0
nii = load_nii('b0_map.nii');
img = nii.img;
img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 1;
b0 = imresize(img_1,2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii,'b0.nii');

% DTI_ADC
nii=load_nii('dti_adc_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 1;
DTI_ADC=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_ADC);
save_nii(nii,'DTI_ADC.nii');

% DTI_FA
nii=load_nii('dti_fa_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
DTI_FA=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(DTI_FA);
save_nii(nii,'DTI_FA.nii');

% fiber_ratio
nii=load_nii('fiber_ratio_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
fiber=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(fiber);
save_nii(nii,'fiber.nii');


% restricted_1_ratio;
nii=load_nii('restricted_ratio_1_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
restricted_1=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_1);
save_nii(nii,'restricted_1.nii');

% restricted_2_ratio;
nii=load_nii('restricted_ratio_2_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
restricted_2=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(restricted_2);
save_nii(nii,'restricted_2.nii');

% hindered_ratio
nii=load_nii('hindered_ratio_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
hindered=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(hindered);
save_nii(nii,'hindered.nii');

% water_ratio
nii=load_nii('water_ratio_map.nii');
img=nii.img;
img_1 = imgaussfilt(img,0.5); % gussian filter with scalar value of sigma of 1;
water=imresize(img_1,2); % interpolate iamgew with bicubic method
nii=make_nii(water);
save_nii(nii,'water.nii');



disp('Generate DBSI Results: Completed!');
















