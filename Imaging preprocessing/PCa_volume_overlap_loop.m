%% to calculate the overlap area of DTI and DBSI map;
% created by yzz on 06/7/2016;
% calculate total prostate voulme & mean value;
% calculate the overlapping degree of DTI- and DBSI-defined PCa for the whole prostate;

function output = PCa_volume_overlap() 
                  
   PCa_data = ls('PCa_*');
   n = length(PCa_data);

   for i = 1:n
       folder = n{i};
       fprintf('Processing folder %s\n', folder);

         % read nii files;
         mask = load_nii(fullfile(folder,'mask.nii'));
         ADC = load_nii(fullfile(folder,'DTI_ADC.nii'));
         restricted = load_nii(fullfile(folder'restricted.nii'));
         mask_1 = mask.img;
         ADC_1 = ADC.img;
         restricted_1 = restricted.img;
         map = double(mask_1).*ADC_1;
         index = find(map>0);
         
         % to calculate whole prostate volumes & values;
         prostate_volume(i) = length(index);
         prostate_volume_mm(i) = prostate_volume*4/1000; %to calculate the actual volume, resoluion of image is 1x1x4 mm3;
         prostate_ADC_mean(i) = mean(map(index));
         prostate_ADC_stdv(i) = std(map(index));

         % to calculate PCa volumes & vaules based on ADC;
         index_1 = find(map <= 1.08 & map > 0);
         PCa_ADC = map(index_1);
         PCa_ADC_volume(i) = length(PCa_ADC);
         PCa_ADC_volume_mm(i) = PCa_ADC_volume*4/1000; %to calculate the actual volume, resoluion of image is 1x1x4 mm3; 
         PCa_ADC_mean(i) = mean(PCa_ADC);
         PCa_ADC_stdv(i) = std(PCa_ADC);
         PCa_ADC_percentage(i) = PCa_ADC_volume / prostate_volume;

         % to calculate PCa volumes & values based on DBSI;
         map_1 = double(mask_1).*restricted_1;
         index_2 = find(map_1 > 0.05 & map_1 < 1);
         PCa_DBSI = map_1(index_2);
         PCa_DBSI_volume(i) = length(PCa_DBSI);
         PCa_DBSI_volum_mm(i) = PCa_DBSI_volume*4/1000; %to calculate the actual volume, resoluion of image is 1x1x4 mm3;
         PCa_DBSI_mean(i) = mean(PCa_DBSI);
         PCa_DBSI_stdv(i) = std(PCa_DBSI);
         PCa_DBSI_percentage(i) = PCa_DBSI_volume / prostate_volume;


         % to calculate the overlaping degree between DBSI and DTI defined PCa for the whole prosate;
         overlap_ratio = length(intersect(index_1, index_2)) / length(union(index_1, index_2));

         output = [PCa_ADC_percentage(i) PCa_DBSI_percentage(i) overlap_ratio(i) prostate_volume(i) prostate_volume_mm(i) PCa_ADC_volume(i) PCa_DBSI_volume(i) PCa_DBSI_volum_mm(i) prostate_ADC_mean(i) prostate_ADC_stdv(i) PCa_ADC_mean(i) PCa_ADC_stdv(i) PCa_DBSI_mean(i) PCa_DBSI_stdv(i)];
         
   end
         
         filename = 'PCa_volume_overlap.xlsx';
         xlswrite(filename,output);


         disp('program end');
end