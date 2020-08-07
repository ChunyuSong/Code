listoffoldernames = {'dbsi01_01','dbsi01_03','dbsi01_05','dbsi01_02','dbsi01_04','dbsi02_01','dbsi02_02','dbsi02_03','dbsi02_04','dbsi02_05','dbsi03_01','dbsi03_02','dbsi03_03','dbsi03_04','dbsi03_05','dbsi04_01','dbsi04_02','dbsi04_03','dbsi04_04','dbsi04_05'};
numberoffolders = length(listoffoldernames);

for k=1:numberoffolders
    thisfolder = listoffoldernames{k};
    fprintf('Processing folder %s\n', thisfolder);
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_adc_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_adc_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_axial_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_axial_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_radial_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_radial_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber_ratio_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber_ratio_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber1_axial_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber1_axial_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber1_radial_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber1_radial_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber1_fa_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\fiber1_fa_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_fa_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\dti_fa_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\hindered_ratio_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\hindered_ratio_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\restricted_ratio_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\restricted_ratio_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\water_ratio_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\water_ratio_map_blur.nii'));
    
    nii = load_nii(fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\restricted_adc_map.nii'));
    img = nii.img;
    img_1 = imgaussfilt(img,1); % gussian filter with scalar value of sigma of 2;
    b0 = imresize(img_1,2); % interpolate iamge with bicubic method
    nii = make_nii(b0);
    save_nii(nii,fullfile(thisfolder,'\DBSI_results_0.3_0.3_2_2\restricted_adc_map_blur.nii'));
end
