(* ::Package:: *)


function output = resize_images _PCa _comp()  

    list = cellstr(ls('*_*'));

    for i = 1:length(list)
           folder = list{i};
           if length(folder) > 67
               break;
           end
           fprintf('Processing folder % s \n', folder);

           % b0
           loadingNifti(folder, 'b0_map.nii');
 
           % model_v _map
           loadingNifti(folder, 'model_v _map.nii');   
           
           % dti_axial _map
           loadingNifti(folder, 'dti_axial _map.nii');
                     
           % dti_radial _map
           loadingNifti(folder, 'dti_radial _map.nii');
           
           
           % DTI_ADC
           loadingNifti(folder, 'dti_adc _map.nii');
           
           
           % DTI_FA
           loadingNifti(folder, 'dti_fa _map.nii');
           
           
           
           % dti_dirx _map
           loadingNifti(folder, 'dti_dirx _map.nii');
           
           % dti_diry _map
           loadingNifti(folder, 'dti_diry _map.nii');
           
           % dti_dirz _map
           loadingNifti(folder, 'dti_dirz _map.nii');
           
           % dti_rgba _map
           loadingNifti(folder, 'dti_rgba _map.nii');
           
           % dti_rgba _map _itk
           loadingNifti(folder, 'dti_rgba _map _itk.nii');
           
           
           % dti_r _map
           loadingNifti(folder, 'dti_r _map.nii');
           
           
           % dti_g _map
           loadingNifti(folder, 'dti_g _map.nii');
           
           
           % dti_b _map
           loadingNifti(folder, 'dti_b _map.nii');
           
           
           % fiber_ratio
           loadingNifti(folder, 'fiber_ratio _map.nii');
           
           
           % fiber1_axial _map
           loadingNifti(folder, 'fiber1_axial _map.nii');
           
           
           % fiber1_radial _map
           loadingNifti(folder, 'fiber1_radial _map.nii');
           
           
           
           % fiber1_fa _map
           loadingNifti(folder, 'fiber1_fa _map.nii');
           
           
           
           % fiber1_fiber _ratio _map
           loadingNifti(folder, 'fiber1_fiber _ratio _map.nii');
           
           
           % fiber1_rgba _map
           loadingNifti(folder, 'fiber1_rgba _map.nii');
           
           
           % fiber1_dirx _map
           loadingNifti(folder, 'fiber1_dirx _map.nii');
           
           
           % fiber1_diry _map
           loadingNifti(folder, 'fiber1_diry _map.nii');
           
           
           % fiber1_dirz _map
           loadingNifti(folder, 'fiber1_dirz _map.nii');
           
           
           % fiber2_axial _map
           loadingNifti(folder, 'fiber2_axial _map.nii');
           
           
           % fiber2_radial _map
           loadingNifti(folder, 'fiber2_radial _map.nii');
           
           % fiber2_fa _map
           loadingNifti(folder, 'fiber2_fa _map.nii');
           
           % fiber2_fiber _ratio _map
           loadingNifti(folder, 'fiber2_fiber _ratio _map.nii');
           
           % fiber2_dirx _map
           loadingNifti(folder, 'fiber2_dirx _map.nii');
           
           
           % fiber2_diry _map
           loadingNifti(folder, 'fiber2_diry _map.nii');
           
           % fiber2_dirz _map
           loadingNifti(folder, 'fiber2_dirz _map.nii');
           
           
           % restricted_ratio _ 1;
           loadingNifti(folder, 'restricted_ratio _ 1_map.nii');
           
           % restricted_adc _ 1;
           loadingNifti(folder, 'restricted_adc _ 1_map.nii');
           
           
           % restricted_ratio _ 2;
           loadingNifti(folder, 'restricted_ratio _ 2_map.nii');
           
           
           % restricted_adc _ 2;
           loadingNifti(folder, 'restricted_adc _ 2_map.nii');
           
           
           % hindered_ratio
           loadingNifti(folder, 'hindered_ratio _map.nii');
           
           
           % hindered_adc
           loadingNifti(folder, 'hindered_adc _map.nii');
           
           
           % water_ratio
           loadingNifti(folder, 'water_ratio _map.nii');
           
           
           % water_adc
           loadingNifti(folder, 'water_adc _map.nii');
           
           % iso_adc
           loadingNifti(folder, 'iso_adc _map.nii');
           
           
           % fraction_rgba _map
           loadingNifti(folder, 'fraction_rgba _map.nii');
           
           
           % fiber1_rgba _map _itk
           loadingNifti(folder, 'fiber1_rgba _map _itk.nii');
       
   end
        disp('Generate DHISTO Results: Completed!');
        
end

function loadingNifti(folder, input)

    filename = fullfile(folder,'DHISTO_results _ 0.1_ 0.1_ 0.8_ 0.8_ 2.3_ 2.3',input);
    nii = load_nii(filename);
    img = nii.img;
    b0 = imresize(img,2); % interpolate iamge with bicubic neighbor method
    nii = make_nii(b0);
    save_nii(nii,filename);

end

    
        
     


















