(* ::Package:: *)

clear all;
maps = {'b0_map','dti_adc _map','dti_fa _map','dti_axial _map','dti_radial _map','iso_adc _map','restricted_ratio _ 1_map','restricted_ratio _ 2_map','hindered_ratio _map','water_ratio _map','fiber_ratio _map'};
dirDBSI = ['Z:\Zezhong_Ye\Prostate_Cancer_Project _Shanghai\2018 _ 08_ 12\6630_JR100734\PELVIS_CHANGHAI_ 20180813_ 092531_ 557000\DBSI_results_ 0.1_ 0.1_ 0.8_ 0.8_ 2.3_ 2.3'];
% savePath = ['Z:\Zezhong_Ye\Prostate_Cancer_Project _Shanghai\2018 _ 08_ 12\6630_JR100734\PELVIS_CHANGHAI_ 20180813_ 092531_ 557000\DBSI_results_ 0.1_ 0.1_ 0.8_ 0.8_ 2.3_ 2.3'];
% if ~exist(savePath, 'dir');
  % mkdir(savePath);
% end;

% generate DWI_ 1500 image
filename_adc = fullfile(dirDBSI,'dti_adc _map.nii');
filename_b0 = fullfile(dirDBSI,'b0_map.nii');
adc = load_nii(filename_adc);
ADC = adc.img;
b0 = load_nii(filename_b0);
B0 = b0.img;
DWI = B0.*exp(-3.0*ADC);
DWI = flipdim(DWI,2); 
DWI = imresize(DWI,2); 
nii = make_nii(DWI);
save_nii(nii,'dwi_ 3000.nii');

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


