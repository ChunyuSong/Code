% generate DWI_1500 image
nii = load_nii('dti_adc_map.nii');
adc = nii.img;
b0 = load_nii('b0_map.nii');
b0 = b0.img;
dwi = b0.*exp(-1.5*adc);
dwi = flipdim(dwi, 2); % flip iamges vertically
dwi_1500 = imresize(dwi,2); % interpolate iamgew with bicubic method
nii = make_nii(dwi_1500);
save_nii(nii,'dwi_1500.nii');