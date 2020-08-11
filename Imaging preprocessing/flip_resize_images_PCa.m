(* ::Package:: *)



% b0
nii = load_nii('pca_ 001_ 5&4_. nii');
img = nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
nii = make_nii(b0);
save_nii(nii,'b0.nii');

% DTI_ADC
nii = load_nii('dti_adc _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_ADC=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(DTI_ADC);
save_nii(nii,'DTI_ADC.nii');

% DTI_FA
nii = load_nii('dti_fa _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
DTI_FA=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(DTI_FA);
save_nii(nii,'DTI_FA.nii');

% fiber_ratio
nii = load_nii('fiber_ratio _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
fiber=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(fiber);
save_nii(nii,'fiber_ratio.nii');

% restricted_ratio _ 1;
nii = load_nii('restricted_ratio _ 1_map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
restricted=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(restricted);
save_nii(nii,'restricted_ratio _ 1.nii');

% restricted_ratio _ 2;
nii = load_nii('restricted_ratio _ 2_map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
restricted=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(restricted);
save_nii(nii,'restricted_ratio _ 2.nii');

% hindered_ratio
nii = load_nii('hindered_ratio _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
hindered=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(hindered);
save_nii(nii,'hindered_ratio.nii');

% water_ratio
nii = load_nii('water_ratio _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(water);
save_nii(nii,'water_ratio.nii');

% iso_adc
nii=load_nii('iso_adc _map.nii');
img=nii.img;
img_ 1 = fliplr(img); % flip iamges vertically
water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
nii=make_nii(water);
save_nii(nii,'iso_adc.nii');

disp('Generate DBSI Results: Completed!');

















