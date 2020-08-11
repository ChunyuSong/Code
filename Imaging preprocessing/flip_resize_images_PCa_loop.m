(* ::Package:: *)

%% to calculate the overlap area of DTI and DBSI map;

function output = flip_resize _images _PCa _loop  


                 
   list = cellstr(ls('P*'));

   for i = 1:length(list)
       folder = list{i};
       if length(folder) > 29
           break;
       end
       fprintf('Processing folder % s \n', folder);

       % b0
        nii=load_nii(fullfile(folder,'b0_map.nii'));
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,[folder filesep 'b0.nii']);

        % DTI_ADC
        nii=load_nii(fullfile(folder,'dti_adc _map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        DTI_ADC=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(DTI_ADC);
        save_nii(nii,[folder filesep 'DTI_ADC.nii']);

        % DTI_FA
        nii=load_nii(fullfile(folder,'dti_fa _map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        DTI_FA=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(DTI_FA);
        save_nii(nii,[folder filesep 'DTI_FA.nii']);

        % fiber_ratio
        nii=load_nii(fullfile(folder,'fiber_ratio _map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        fiber=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(fiber);
        save_nii(nii,[folder filesep 'fiber_ratio.nii']);

        % restricted_ratio _ 1;
        nii=load_nii(fullfile(folder,'restricted_ratio _ 1_map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,[folder filesep 'restricted_ratio _ 1.nii']);

        % restricted_ratio _ 2;
        nii=load_nii(fullfile(folder,'restricted_ratio _ 2_map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,[folder filesep 'restricted_ratio _ 2.nii']);

        % hindered_ratio
        nii=load_nii(fullfile(folder,'hindered_ratio _map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        hindered=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(hindered);
        save_nii(nii,[folder filesep 'hindered_ratio.nii']);

        % water_ratio
        nii=load_nii(fullfile(folder,'water_ratio _map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,[folder filesep 'water_ratio.nii']);

        % iso_adc
        nii=load_nii(fullfile(folder,'iso_adc _map.nii'));
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,[folder filesep 'iso_adc.nii']);


   end
   
end
