    function [color_markers_temp,image_data,Red,Green,Blue,roi_count] = select_roi(imdata,roi_count,image_count,numberOfColorBands,color_markers_temp,image_data,Red,Green,Blue)

     imdata = imgaussfilt(imdata, 0.5);

%      imdata = adapthisteq(imdata, 'Distribution','rayleigh');
    
    image_data(:,:,:,image_count) = imdata;

    if numberOfColorBands > 1

        Red(:,:,1,image_count) = imdata(:,:,1);
        Green(:,:,1,image_count) = imdata(:,:,2);
        Blue(:,:,1,image_count) = imdata(:,:,3);
    
        red_sample = imdata(:,:,1);
        green_sample = imdata(:,:,2);
        blue_sample = imdata(:,:,3);
    else
        Red(:,:,1,image_count) = imdata(:,:,1);
        Green(:,:,1,image_count) = imdata(:,:,1);
        Blue(:,:,1,image_count) = imdata(:,:,1);

        red_sample = imdata(:,:,1);
        green_sample = imdata(:,:,1);
        blue_sample = imdata(:,:,1);
    end
    fig = figure(2);
    
    figure(fig); imshow(imdata);
    title('Choose ROI(s) in Raw Image')

    cmap = colormap(fig);
    
    roi_count = roi_count + 1;
     
    while true
        hfree=imfreehand(gca); 
  
        sample_regions(:,:,roi_count) = createMask(hfree);
        
        
         color_markers_1 = red_sample(sample_regions(:,:,roi_count));
         color_markers_2 = green_sample(sample_regions(:,:,roi_count));
         color_markers_3 = blue_sample(sample_regions(:,:,roi_count));
         
         size(color_markers_1)
         
         if roi_count == 1
             color_markers_temp(:,1) = color_markers_1;
             color_markers_temp(:,2) = color_markers_2;
             color_markers_temp(:,3) = color_markers_3;
         else
        
            color_markers_temp = [color_markers_temp;[color_markers_1 color_markers_2 color_markers_3]];

         end
        
    % Construct a questdlg with three options to further continue segmentation or
    % testing
    choice = questdlg('Do you want to select roi again?', ...
    'Question', ...
    'Yes','No','Yes');
    % Handle response
    switch choice
    case 'Yes'    
      roi_count = roi_count + 1;
      continue;
    case 'No'  
      break;
    end
    end  