(* ::Package:: *)

% b0
nii = load_nii('b0_map.nii');
A = nii.img;
A1 = flipdim(A, 2) ;  % flip iamges vertically
b0 = imresize(A1, 2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii, 'b0_map.nii');

% DTI_ADC
nii = load_nii('dti_adc _map.nii');
B = nii.img;
B1 = flipdim(B, 2);
DTI_ADC = imresize(B1,2);
nii = make_nii(DTI_ADC);
save_nii(nii, 'dti_adc _map.nii');

% DTI_FA
nii = load_nii('dti_fa _map.nii');
B = nii.img;
B1 = flipdim(B,2);
DTI_FA = imresize(B1,2);
nii = make_nii(DTI_FA);
save_nii(nii, 'dti_fa _map.nii');


% Fiber_ratio
nii = load_nii('fiber_ratio _map.nii');
B = nii.img;
B1 = flipdim(B, 2) ;
BPH = imresize(B1, 2); 
nii = make_nii(BPH);
save_nii(nii, 'fiber_ratio _map.nii');

% restricted_ratio _ 1
nii = load_nii('restricted_ratio _ 1_map.nii');
B = nii.img;
B1 = flipdim(B, 2); 
Lymphocytes = imresize(B1, 2);
nii = make_nii(Lymphocytes);
save_nii(nii, 'restricted_ratio _ 1_map.nii');

% restricted_ratio _ 2
nii = load_nii('restricted_ratio _ 2_map.nii');
B = nii.img;
B1 = flipdim(B, 2); 
Lymphocytes = imresize(B1, 2);
nii = make_nii(Lymphocytes);
save_nii(nii, 'restricted_ratio _ 2_map.nii');

% hindered_ratio
nii = load_nii('hindered_ratio _map.nii');
B = nii.img;
B1 = flipdim(B, 2); 
PCa = imresize(B1, 2);
nii = make_nii(PCa);
save_nii(nii, 'hindered_ratio _map.nii');

% water_ratio
nii = load_nii('water_ratio _map.nii');
B = nii.img;
B1 = flipdim(B, 2) ;
prostate = imresize(B1, 2); 
nii = make_nii(prostate);
save_nii(nii, 'water_ratio _map.nii');

% generate DWI_ 1500 image
nii = load_nii('dti_adc _map.nii');
adc = nii.img;
b0 = load_nii('b0_map.nii');
b0 = b0.img;
c = b0.*exp(-1.5*adc);
nii = make_nii(c);
save_nii(nii, 'dwi_ 1500.nii');

disp('interpolation complete!!!!')


















