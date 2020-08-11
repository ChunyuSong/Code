(* ::Package:: *)

% b0
nii = load_nii('dti_me _denoised _combinedecho.nii');
A = nii.img;
b0 = imresize(A,2); % interpolate iamge with bicubic method
nii = make_nii(b0);
save_nii(nii,'dbsi.nii');

















