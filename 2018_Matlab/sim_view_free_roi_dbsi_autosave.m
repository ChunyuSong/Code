function [] = sim_view_free_roi_dbsi_autosave(sim_spectrum_adc_mat,sim_input_mat)
% Input : 
% sim_spectrum_adc_mat : DBSIClassData.mat file 
% sim_input_mat : dbsi_input.mat file 
 
% Output : To display spectrum of user selected region in image


%% input arguments validation
if nargin < 2 
    sim_input_mat=[];
    if nargin < 1 
        sim_spectrum_adc_mat=[];
    end
end

sim_spectrum_adc_mat=[];
sim_input_mat=[];

if isempty(sim_spectrum_adc_mat) || ~exist(sim_spectrum_adc_mat,'file')
    [file, path] = uigetfile('*.mat', 'Choose DBSIClassData file ...');
    if path == 0 
        error('you didn''t choose DBSIClassData file, program terminated!');
    end
    sim_spectrum_adc_mat = fullfile(path,file);
    if  ~strcmp(sim_spectrum_adc_mat(end-3:end),'.mat') || ~exist(sim_spectrum_adc_mat,'file')
        error('you didn''t choose DBSIClassData file or the file doesn''t exist, program terminated!');
    end
end

if isempty(sim_input_mat) || ~exist(sim_input_mat,'file')
    [file, path] = uigetfile('*.mat', 'Choose dbsi_input file ...');
    if path == 0 
        error('you didn''t choose dbsi_input file, program terminated!');
    end
    sim_input_mat = fullfile(path,file);
    if  ~strcmp(sim_input_mat(end-3:end),'.mat') || ~exist(sim_input_mat,'file')
        error('you didn''t choose dbsi_input file or the file doesn''t exist, program terminated!');
    end
end    

%% load file(s)
    load(sim_spectrum_adc_mat);
    load(sim_input_mat);

    if ~exist('cRawData','var') 
        error('Missing Variable: input DBSIClassData file, program terminated!');
    end
    
    if ~exist('dbsi_data','var') || ~exist('b_value','var')
        error('Missing Variables: input dbsi_input file, program terminated!');
    end
    
    map_dimensions = [cRawData.sImageHeader.image_size, cRawData.sImageHeader.ns];
    roi_index=sub2ind(map_dimensions,cRawData.DBSI_aRoiIndex(1,:),cRawData.DBSI_aRoiIndex(2,:),cRawData.DBSI_aRoiIndex(3,:));
    data_spectrum = zeros([map_dimensions,size(cRawData.DBSI_aIsoSpecdata,1)]);
    data_spectrum = reshape(data_spectrum,[],size(cRawData.DBSI_aIsoSpecdata,1));

    data_spectrum(roi_index,:) = cRawData.DBSI_aIsoSpecdata';
    data_spectrum = reshape(data_spectrum,[map_dimensions,size(cRawData.DBSI_aIsoSpecdata,1)]);


    adc = cRawData.DBSI_iIsoSpecGrid;
    
    
    dbsi_data = permute(dbsi_data,[1 2 4 3]);
    
    num_slices = size(data_spectrum,3);
    num_bvalue = size(dbsi_data,4);
    
                               
    str1=sprintf('Please input slice number of %d\n',num_slices);
    str2 = ' total slices';
    str1 = strcat(str1,str2);
    str3=sprintf('Please input bvalue number of %d\n',num_bvalue);
    str4 = ' total bvalues';
    str3 = strcat(str3,str4);        
    prompt = {str1,str3};
    dlg_title = 'Slice & Bval Input';
    num_lines = 1;
    defaultans = {'1','1'};
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    if isempty(answer)
        slice_num = 1;
        bvalue_num = 1;
    else
        slice_num = str2double(answer(1));
        bvalue_num =  str2double(answer(2));
    end

    img = dbsi_data(:,:,slice_num, bvalue_num);

    
%%  display input image and spectrum
    h_fig_img=figure(1);  
    done=false;
    while ~done              
          figure(h_fig_img)
          imshow(img,[0 max(img(:))]);
          impixelinfo;             
          title('Spectrum Image');
          set(h_fig_img,'Pointer','crosshair');

          if gcf==h_fig_img
            ax=gca; 
            h_free=imfreehand(ax);
            try
                key = get(h_fig_img,'CurrentCharacter');
            catch Exception
                break;
            end
            if key==27
                close all;
                break;
            else
                set(h_fig_img,'CurrentCharacter','@');
                waitfor(h_fig_img,'CurrentCharacter');
                try
                    key = get(h_fig_img,'CurrentCharacter');
                    BW = createMask(h_free); 
                    [row,col] = find(BW);
                catch Exception
                    break;
                end
                for i=1:length(adc)
                    data(:,:,i)=squeeze(data_spectrum(:,:,slice_num,i)).*BW;
                    [~,~,val]=find(data(:,:,i));
                    Avg_data_spectrum(i) = sum(val)/length(find(BW));
                end              
                for i=1:length(b_value)                                     
                    bdata(:,:,i)=squeeze(dbsi_data(:,:,slice_num,i)).*BW;
                    [~,~,b_val]=find(bdata(:,:,i));
                    Avg_signal(i) = sum(b_val)/length(find(BW));                    
                end              
                if key ~=27                                                              
                    h_fig_spect=figure(2);
                    plot(adc,Avg_data_spectrum,'linewidth',5);
                    grid off;
                    box off;
                    xlim([0 0.003]);
                    %xlabel('Isotropic ADC');ylabel('Amplitudes'); title('D-Histo Spectrum Analysis'); 
                    set(gca,'fontsize',30, 'fontweight','bold','linewidth',6);
                    set(gca,'TickDir','out');
                    set(gca,'XTick',0:0.0005:0.003);
                    hold on;  
                end
                legend('region a','region b','region c','region d'); 
            end
         end       
    end
end