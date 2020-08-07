%% DBSI prostate processing by yzz at 8/13/2018;

clear all;
maps = {'b0_map','dti_adc_map','dti_fa_map','dti_axial_map','dti_radial_map','iso_adc_map','restricted_ratio_1_map','restricted_ratio_2_map','hindered_ratio_map','water_ratio_map','fiber_ratio_map'};
dirDBSI = ['Z:\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\2018_08_12\6630_JR100734\PELVIS_CHANGHAI_20180813_092531_557000\DBSI_results_0.1_0.1_0.8_0.8_2.3_2.3'];
%savePath = ['Z:\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\2018_08_12\6630_JR100734\PELVIS_CHANGHAI_20180813_092531_557000\DBSI_results_0.1_0.1_0.8_0.8_2.3_2.3'];
%if ~exist(savePath, 'dir');
  %mkdir(savePath);
%end;

% generate DWI_1500 image
filename_adc = fullfile(dirDBSI,'dti_adc_map.nii');
filename_b0 = fullfile(dirDBSI,'b0_map.nii');
adc = load_nii(filename_adc);
ADC = adc.img;
b0 = load_nii(filename_b0);
B0 = b0.img;
DWI = B0.*exp(-3.0*ADC);
DWI = flipdim(DWI,2); 
DWI = imresize(DWI,2); 
nii = make_nii(DWI);
save_nii(nii,'dwi_3000.nii');

for i = 1:numel(maps);
    map = maps{i};
    img = char(map);
    filename = fullfile(dirDBSI,[img,'.nii']);
    image = load_nii(filename);
    MRI = image.img;
    MRI = flipdim(MRI,2); 
    MRI = imresize(MRI,2); 
    nii = make_nii(MRI);
    file_name = [img,'.nii'];
    save_nii(nii,file_name);
end;

disp('prostate processing: Completed!');


