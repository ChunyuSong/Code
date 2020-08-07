%% flip the DBSI.nii maps by yzz on 04/13/2016
%% edited on 05/05/2016 interpolate maps by twice
%% edited by JLin on 08/10/2017 for complete dbsi parameters
%% edited by JLin on 09/15/2017 to subsitute bicubic interpolation to nearest neighbor interpolation
function output = resize_images_PCa_comp()  

    list = cellstr(ls('*_*'));

    for i = 1:length(list)
           folder = list{i};
           if length(folder) > 67
               break;
           end
           fprintf('Processing folder %s\n', folder);

           % b0
           loadingNifti(folder, 'b0_map.nii');
 
           % model_v_map
           loadingNifti(folder, 'model_v_map.nii');   
           
           % dti_axial_map
           loadingNifti(folder, 'dti_axial_map.nii');
                     
           % dti_radial_map
           loadingNifti(folder, 'dti_radial_map.nii');
           
           
           % DTI_ADC
           loadingNifti(folder, 'dti_adc_map.nii');
           
           
           % DTI_FA
           loadingNifti(folder, 'dti_fa_map.nii');
           
           
           
           % dti_dirx_map
           loadingNifti(folder, 'dti_dirx_map.nii');
           
           % dti_diry_map
           loadingNifti(folder, 'dti_diry_map.nii');
           
           % dti_dirz_map
           loadingNifti(folder, 'dti_dirz_map.nii');
           
           % dti_rgba_map
           loadingNifti(folder, 'dti_rgba_map.nii');
           
           % dti_rgba_map_itk
           loadingNifti(folder, 'dti_rgba_map_itk.nii');
           
           
           % dti_r_map
           loadingNifti(folder, 'dti_r_map.nii');
           
           
           % dti_g_map
           loadingNifti(folder, 'dti_g_map.nii');
           
           
           % dti_b_map
           loadingNifti(folder, 'dti_b_map.nii');
           
           
           % fiber_ratio
           loadingNifti(folder, 'fiber_ratio_map.nii');
           
           
           % fiber1_axial_map
           loadingNifti(folder, 'fiber1_axial_map.nii');
           
           
           % fiber1_radial_map
           loadingNifti(folder, 'fiber1_radial_map.nii');
           
           
           
           % fiber1_fa_map
           loadingNifti(folder, 'fiber1_fa_map.nii');
           
           
           
           % fiber1_fiber_ratio_map
           loadingNifti(folder, 'fiber1_fiber_ratio_map.nii');
           
           
           % fiber1_rgba_map
           loadingNifti(folder, 'fiber1_rgba_map.nii');
           
           
           % fiber1_dirx_map
           loadingNifti(folder, 'fiber1_dirx_map.nii');
           
           
           % fiber1_diry_map
           loadingNifti(folder, 'fiber1_diry_map.nii');
           
           
           % fiber1_dirz_map
           loadingNifti(folder, 'fiber1_dirz_map.nii');
           
           
           % fiber2_axial_map
           loadingNifti(folder, 'fiber2_axial_map.nii');
           
           
           % fiber2_radial_map
           loadingNifti(folder, 'fiber2_radial_map.nii');
           
           % fiber2_fa_map
           loadingNifti(folder, 'fiber2_fa_map.nii');
           
           % fiber2_fiber_ratio_map
           loadingNifti(folder, 'fiber2_fiber_ratio_map.nii');
           
           % fiber2_dirx_map
           loadingNifti(folder, 'fiber2_dirx_map.nii');
           
           
           % fiber2_diry_map
           loadingNifti(folder, 'fiber2_diry_map.nii');
           
           % fiber2_dirz_map
           loadingNifti(folder, 'fiber2_dirz_map.nii');
           
           
           % restricted_ratio_1;
           loadingNifti(folder, 'restricted_ratio_1_map.nii');
           
           % restricted_adc_1;
           loadingNifti(folder, 'restricted_adc_1_map.nii');
           
           
           % restricted_ratio_2;
           loadingNifti(folder, 'restricted_ratio_2_map.nii');
           
           
           % restricted_adc_2;
           loadingNifti(folder, 'restricted_adc_2_map.nii');
           
           
           % hindered_ratio
           loadingNifti(folder, 'hindered_ratio_map.nii');
           
           
           % hindered_adc
           loadingNifti(folder, 'hindered_adc_map.nii');
           
           
           % water_ratio
           loadingNifti(folder, 'water_ratio_map.nii');
           
           
           % water_adc
           loadingNifti(folder, 'water_adc_map.nii');
           
           % iso_adc
           loadingNifti(folder, 'iso_adc_map.nii');
           
           
           % fraction_rgba_map
           loadingNifti(folder, 'fraction_rgba_map.nii');
           
           
           % fiber1_rgba_map_itk
           loadingNifti(folder, 'fiber1_rgba_map_itk.nii');
       
   end
        disp('Generate DHISTO Results: Completed!');
        
end

function loadingNifti(folder, input)

    filename = fullfile(folder,'DHISTO_results_0.1_0.1_0.8_0.8_2.3_2.3',input);
    nii = load_nii(filename);
    img = nii.img;
    b0 = imresize(img,2); % interpolate iamge with bicubic neighbor method
    nii = make_nii(b0);
    save_nii(nii,filename);

end

    
        
     


















