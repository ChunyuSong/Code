function [] = distance_calc(image_count,color_markers,color_markers_temp,image_data,Red,Green,Blue,numberOfColorBands)

distance = zeros([size(squeeze(Red(:,:,1,image_count))),1,image_count]);

label_final = zeros([size(squeeze(Red(:,:,1,image_count))),1,image_count]);

%% choose stain type and nearest neighbor classification
for count = 1:image_count   

    R = double(squeeze(Red(:,:,1,count)));
    G = double(squeeze(Green(:,:,1,count)));
    B = double(squeeze(Blue(:,:,1,count)));
    
    if numberOfColorBands > 1
        distance(:,:,1,count) = ( (R - color_markers(1,1)).^2 + (G - color_markers(1,2)).^2 + (B - color_markers(1,3)).^2).^0.5;

        distance_roi = ((double(color_markers_temp(:,1)) - color_markers(1,1)).^2 + double((color_markers_temp(:,2)) - color_markers(1,2)).^2 + double((color_markers_temp(:,3)) - color_markers(1,3)).^2).^0.5;
    else
       distance(:,:,1,count) = ( ((R - color_markers(1,1)).^2 + (G - color_markers(1,2)).^2 + (B - color_markers(1,3)).^2))/3.^0.5;

       distance_roi = (((double(color_markers_temp(:,1)) - color_markers(1,1)).^2 + double((color_markers_temp(:,2)) - color_markers(1,2)).^2 + double((color_markers_temp(:,3)) - color_markers(1,3)).^2))/3.^0.5;
    end

    distance_roi = uint8(mat2gray(distance_roi).*63);

%% steps to generate binary image highlighting  object or region of interest

    label = squeeze(distance(:,:,1,count));
    
    label_final(:,:,1,count) = mat2gray(label);
    [X(:,:,1,count),map] = gray2ind(squeeze(label_final(:,:,1,count)));    
    
end
figure;
montage(image_data)
title('Raw Image');

figure;
montage(X,[])
colormap(jet)
title('Distance Map');


while true

    figx = figure();
    hist(double(distance_roi),64);
    xlabel('Distance Index'),ylabel('Frequency')
    title('Histogram of Distance Index')
    waitfor(figx);
    
    prompt = {'Please an index between 0 & 63'};
    dlg_title = 'Input i and j max index for segmented images';
    num_lines = 1;
    defaultans = {'10'};
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);


    if isempty(answer) || str2double(answer) > 63 || str2double(answer) < 0
        error('Invalid values for index');
    else
        index = str2double(answer);
    end    

str_1 = [];

for count = 1:image_count
    
        if numberOfColorBands > 1
            BW(:,:,1,count) = roicolor(squeeze(X(:,:,1,count)),[0:index]);
        else
%             BW(:,:,1,count) = squeeze(mat2gray(Red(:,:,1,count)).*63) > ((mat2gray(color_markers(1,1)).*63)-double(index));
              BW(:,:,1,count) = squeeze((Red(:,:,1,count)).*63) > color_markers(1,1).*63-double(index);

        end
        
%         BW(:,:,1,count) = imfill(squeeze(BW(:,:,1,count)),'holes');
    
%% calculate area corresponding to chosen object or region of interest

       area(:,:,1,count) = bwarea(BW(:,:,1,count)) * 100/numel(BW(:,:,1,count));
       
       if numberOfColorBands > 1

         BW(:,:,1,count) = bwareaopen(BW(:,:,1,count), 10);

         area(:,:,1,count) = bwarea(BW(:,:,1,count))* 100/numel(BW(:,:,1,count));
       else
            test = (find(R>0));
            if length(test) == 0
                test = numel(BW(:,:,1,count));
            else 
                test = length(test);
            end

            area(:,:,1,count) = bwarea(BW(:,:,1,count))* 100/test;
            
            if test < (0.7*6084)
                area(:,:,1,count) = 0;
            end
       end
    
       str_1 = [str_1 ' ' num2str(area(:,:,1,count)) '%'];
    
end

figBW = figure;
montage(BW)
title('Binary Image')
waitfor(figBW);

    choice = questdlg('Do you want to change index?', ...
    'Question', ...
    'Yes','No','Yes');
    % Handle response
    switch choice
    case 'Yes'    
      continue;
    case 'No' 
      save('color_marker.mat','color_markers','index');  
      break;
    end 
end

%% display results
figure_handle = figure()
ha(1)= subplot(121);
hraw = montage(image_data);
title('raw image')
MyMontage = get(hraw, 'CData');
imwrite(MyMontage,'raw.tif');

ha(2) = subplot(122);
hbw = montage(BW);
MyMontage = get(hbw, 'CData');
imwrite(uint8(255.*MyMontage),'bw.tif');
title('Percent Area of segmentation');
xlabel(['area = ',str_1,'; total area = ',num2str(prod(size(BW(:,:,1,1))))])
