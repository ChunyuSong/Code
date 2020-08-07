%% flip the DBSI.nii maps by yzz on 04/13/2016
%% edited on 05/05/2016 interpolate maps by twice
%% edited by jl on 08/10/2017 for complete dbsi parameters

% b0
nii = load_nii('pca_001_5&4_.nii');
img = nii.img;
img_1 = fliplr(img); % flip iamges vertically
b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
nii = make_nii(b0);
save_nii(nii,'b0.nii');

% DTI_ADC
nii = load_nii('dti_adc_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_ADC=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(DTI_ADC);
save_nii(nii,'DTI_ADC.nii');

% DTI_FA
nii = load_nii('dti_fa_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
DTI_FA=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(DTI_FA);
save_nii(nii,'DTI_FA.nii');

% fiber_ratio
nii = load_nii('fiber_ratio_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
fiber=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(fiber);
save_nii(nii,'fiber_ratio.nii');

% restricted_ratio_1;
nii = load_nii('restricted_ratio_1_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
restricted=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(restricted);
save_nii(nii,'restricted_ratio_1.nii');

% restricted_ratio_2;
nii = load_nii('restricted_ratio_2_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
restricted=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(restricted);
save_nii(nii,'restricted_ratio_2.nii');

% hindered_ratio
nii = load_nii('hindered_ratio_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
hindered=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(hindered);
save_nii(nii,'hindered_ratio.nii');

% water_ratio
nii = load_nii('water_ratio_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(water);
save_nii(nii,'water_ratio.nii');

% iso_adc
nii=load_nii('iso_adc_map.nii');
img=nii.img;
img_1 = fliplr(img); % flip iamges vertically
water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(water);
save_nii(nii,'iso_adc.nii');

disp('Generate DBSI Results: Completed!');

















