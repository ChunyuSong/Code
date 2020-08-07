%% flip the DBSI.nii maps by yzz on 04/13/2016

% b0
nii = load_nii('b0_map.nii');
A = nii.img;
A1 = flipdim(A, 2) ;  % flip iamges vertically
b0 = imresize(A1, 2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii, 'b0_map.nii');

% DTI_ADC
nii = load_nii('dti_adc_map.nii');
B = nii.img;
B1 = flipdim(B, 2);
DTI_ADC = imresize(B1,2);
nii = make_nii(DTI_ADC);
save_nii(nii, 'dti_adc_map.nii');

% DTI_FA
nii = load_nii('dti_fa_map.nii');
B = nii.img;
B1 = flipdim(B,2);
DTI_FA = imresize(B1,2);
nii = make_nii(DTI_FA);
save_nii(nii, 'dti_fa_map.nii');


% Fiber_ratio
nii = load_nii('fiber_ratio_map.nii');
B = nii.img;
B1 = flipdim(B, 2) ;
BPH = imresize(B1, 2); 
nii = make_nii(BPH);
save_nii(nii, 'fiber_ratio_map.nii');

% restricted_ratio_1
nii = load_nii('restricted_ratio_1_map.nii');
B = nii.img;
B1 = flipdim(B, 2); 
Lymphocytes = imresize(B1, 2);
nii = make_nii(Lymphocytes);
save_nii(nii, 'restricted_ratio_1_map.nii');

% restricted_ratio_2
nii = load_nii('restricted_ratio_2_map.nii');
B = nii.img;
B1 = flipdim(B, 2); 
Lymphocytes = imresize(B1, 2);
nii = make_nii(Lymphocytes);
save_nii(nii, 'restricted_ratio_2_map.nii');

% hindered_ratio
nii = load_nii('hindered_ratio_map.nii');
B = nii.img;
B1 = flipdim(B, 2); 
PCa = imresize(B1, 2);
nii = make_nii(PCa);
save_nii(nii, 'hindered_ratio_map.nii');

% water_ratio
nii = load_nii('water_ratio_map.nii');
B = nii.img;
B1 = flipdim(B, 2) ;
prostate = imresize(B1, 2); 
nii = make_nii(prostate);
save_nii(nii, 'water_ratio_map.nii');

% generate DWI_1500 image
nii = load_nii('dti_adc_map.nii');
adc = nii.img;
b0 = load_nii('b0_map.nii');
b0 = b0.img;
c = b0.*exp(-1.5*adc);
nii = make_nii(c);
save_nii(nii, 'dwi_1500.nii');

disp('interpolation complete!!!!')


















