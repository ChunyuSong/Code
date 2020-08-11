(* ::Package:: *)

%%% Display multiple maps in single panel;

clear all;
maps = {'dti_fa _map','restricted_ratio _ 1_map','restricted_ratio _ 2_map','hindered_ratio _map','water_ratio _map','fiber_ratio _map'};
names = {'DTI FA','highly restricted','restricted','hindered','free','anisotropic'};
% dirDBSI = ['C:\Users\zye01\Desktop\GBM\DBSI_results_ 0.2_ 0.2_ 1.5_ 1.5_ 2.5_ 2.5\'];

for i = 1:numel(maps);
    map = maps{i};
    img = char(map);
    name = names{i};
    map_name = char(name);
    subplottight(2,4,i);
    filename = fullfile([img,'.nii']);
    nii = load_nii(filename);
    img = nii.img;
    img = img(:,:,4);
    imshow(img,[0 1],'colormap',jet),colorbar;
    title(map_name);
    hold on;  
end;

subplot(2,4,1);
nii = load_nii('dti_adc _map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 2],'colormap',gray),colorbar;
title('DTI ADC');
hold on;

subplot(2,4,2);
nii = load_nii('dti_fa _map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('DTI FA');
hold on;

subplot(2,4,3);
nii = load_nii('restricted_ratio _ 1_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Higly Restricted');
hold on;

subplot(2,4,4);
nii = load_nii('restricted_ratio _ 2_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Restricted');
hold on;

subplot(2,4,5);
nii = load_nii('hindered_ratio _map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Hindered');
hold on;

subplot(2,4,6);
nii = load_nii('water_ratio _map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Hindered');
hold on;

subplot(2,4,7);
nii = load_nii('fiber_ratio _map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Anisotropic');
hold on;
