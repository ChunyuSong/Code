%% flip the DBSI.nii maps by yzz on 04/13/2016

% b0
nii = load_nii('dti_me_denoised_combinedecho.nii');
A = nii.img;
b0 = imresize(A,2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii,'dbsi.nii');

















