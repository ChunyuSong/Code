(* ::Package:: *)


function output = prostate_cancer()  


                 
   list = cellstr(ls('Patient_*'));

   for i = 1:length(list)
       folder = list{i};
       if length(folder) > 13
           break;
       end
       fprintf('Processing folder % s \n', folder);
       
       % b0
        nii = load_nii('b0_map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'b0.nii');

        % model_v _map
        nii = load_nii('model_v _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'model_v.nii');

        % dti_axial _map
        nii = load_nii('dti_axial _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_axial.nii');

        % dti_radial _map
        nii = load_nii('dti_radial _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_radial.nii');

        % DTI_ADC
        nii=load_nii('dti_adc _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        DTI_ADC=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(DTI_ADC);
        save_nii(nii,'DTI_ADC.nii');

        % DTI_FA
        nii=load_nii('dti_fa _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        DTI_FA=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(DTI_FA);
        save_nii(nii,'DTI_FA.nii');


        % dti_dirx _map
        nii = load_nii('dti_dirx _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_dirx.nii');

        % dti_diry _map
        nii = load_nii('dti_diry _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_diry.nii');

        % dti_dirz _map
        nii = load_nii('dti_dirz _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_dirz.nii');

        % dti_rgba _map
        nii = load_nii('dti_rgba _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_rgba.nii');

        % dti_rgba _map _itk
        nii = load_nii('dti_rgba _map _itk.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_rgba _itk.nii');

        % dti_r _map
        nii = load_nii('dti_r _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_r.nii');

        % dti_g _map
        nii = load_nii('dti_g _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_g.nii');

        % dti_b _map
        nii = load_nii('dti_b _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_b.nii');

        % fiber_ratio
        nii=load_nii('fiber_ratio _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        fiber=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(fiber);
        save_nii(nii,'fiber_ratio.nii');

        % fiber1_axial _map
        nii = load_nii('fiber1_axial _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_axial.nii');

        % fiber1_radial _map
        nii = load_nii('fiber1_radial _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_radial.nii');


        % fiber1_fa _map
        nii = load_nii('fiber1_fa _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_fa.nii');


        % fiber1_fiber _ratio _map
        nii = load_nii('fiber1_fiber _ratio _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_fiber _ratio.nii');


        % fiber1_rgba _map
        nii = load_nii('fiber1_rgba _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_rgba.nii');

        % fiber1_dirx _map
        nii = load_nii('fiber1_dirx _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_dirx.nii');

        % fiber1_diry _map
        nii = load_nii('fiber1_diry _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_diry.nii');

        % fiber1_dirz _map
        nii = load_nii('fiber1_dirz _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_dirz.nii');

        % fiber2_axial _map
        nii = load_nii('fiber2_axial _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_axial.nii');

        % fiber2_radial _map
        nii = load_nii('fiber2_radial _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_radial.nii');


        % fiber2_fa _map
        nii = load_nii('fiber2_fa _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_fa.nii');


        % fiber2_fiber _ratio _map
        nii = load_nii('fiber2_fiber _ratio _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_fiber _ratio.nii');

        % fiber2_dirx _map
        nii = load_nii('fiber2_dirx _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_dirx.nii');

        % fiber2_diry _map
        nii = load_nii('fiber2_diry _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_diry.nii');

        % fiber2_dirz _map
        nii = load_nii('fiber2_dirz _map.nii');
        img = nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_ 1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_dirz.nii');

        % restricted_ratio _ 1;
        nii=load_nii('restricted_ratio _ 1_map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_ratio _ 1.nii');

        % restricted_adc _ 1;
        nii=load_nii('restricted_adc _ 1_map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_adc _ 1.nii');

        % restricted_ratio _ 2;
        nii=load_nii('restricted_ratio _ 2_map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_ratio _ 2.nii');

        % restricted_adc _ 2;
        nii=load_nii('restricted_adc _ 2_map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_adc _ 2.nii');

        % hindered_ratio
        nii=load_nii('hindered_ratio _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        hindered=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(hindered);
        save_nii(nii,'hindered_ratio.nii');

        % hindered_adc
        nii=load_nii('hindered_adc _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        hindered=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(hindered);
        save_nii(nii,'hindered_adc.nii');

        % water_ratio
        nii=load_nii('water_ratio _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'water_ratio.nii');

        % water_adc
        nii=load_nii('water_adc _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'water_adc.nii');

        % iso_adc
        nii=load_nii('iso_adc _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'iso_adc.nii');

        % fraction_rgba _map
        nii=load_nii('fraction_rgba _map.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'fraction_rgba.nii');

        % fiber1_rgba _map _itk
        nii=load_nii('fiber1_rgba _map _itk.nii');
        img=nii.img;
        img_ 1 = fliplr(img); % flip iamges vertically
        water=imresize(img_ 1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'fiber1_rgba _itk.nii');

        % generate DWI_ 1500 image
        nii = load_nii('dti_adc _map.nii');
        adc=nii.img;
        b0=load_nii('b0_map.nii');
        b0=b0.img;
        img=b0.*exp(-1.5*adc);
        img_ 1 = fliplr(img); % flip iamges vertically
        dwi_ 1500=imresize(img_ 1,2,'nearest'); % interpolate iamgew with bi cubic method
        nii = make_nii(dwi_ 1500);
        save_nii(nii,'dwi_ 1500.nii');

        disp('Generate DBSI Results: Completed!');


















