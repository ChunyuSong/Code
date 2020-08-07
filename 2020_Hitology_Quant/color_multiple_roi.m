clear all;

image_count = 1;

roi_count = 0;

default_input_mat = 'Image_data.mat';

if ~exist(default_input_mat,'file')
        [filename, pathname] = uigetfile('*.tif', 'Select first tiled TIFF file');
        if isequal(filename,0)
            error('User selected Cancel');
        else
            fullfilename = fullfile(pathname, filename);
            disp(['User selected ', fullfilename]);
            imdata_resize = imread(fullfilename);
            tile_size = size(imdata_resize);
            imdata_resize = imresize(imdata_resize,0.1);
            idx = strfind(filename,'z0');

            basename = char(filename(1:idx+1));

            [filename, pathname] = uigetfile('*.tif', 'Select last tiled TIFF file');
            if isequal(filename,0)
                error('User selected Cancel');
            else
                fullfilename = fullfile(pathname, filename);
                disp(['User selected ', fullfilename]);
                index_value = char(filename(idx+3:end));
                ivalue = index_value((strfind(index_value, 'i')+1):(strfind(index_value, 'j')-1));                    
                ivalue = str2double(ivalue);
                jvalue = index_value((strfind(index_value, 'j')+1):(strfind(index_value, '.')-1));
                jvalue = str2double(jvalue);
            end

            for i = 1 : ivalue
                 for j = 1 : jvalue
                     if ivalue < 100
                        iindex = num2str(100+i);
                     else
                        iindex = num2str(1000+i);
                     end
                     iindex = iindex(2:end);
                         if jvalue < 100
                            jindex = num2str(100+j);
                         else
                            jindex = num2str(1000+j);
                         end
                         jindex = jindex(2:end);        
                         fullfilename = [basename,'_','i',iindex,'j',jindex,'.tif'];
        
                         imdata_test = imread(fullfilename);
                         imdata_test = imresize(imdata_test,0.1);

                        if j == 1
                             wideImage = imdata_test;
                        else
                        wideImage = [wideImage,imdata_test];
                        end
                 end
                    if i == 1
                        tallImage = wideImage;
                    else
                        tallImage = [tallImage;wideImage];
                    end
                        wideImage = [];      
            end
            save('Image_data.mat','tallImage','tile_size','ivalue','jvalue','basename');
        end
     else

        load(default_input_mat);          
end

while true
    close all;
    hfig = figure(1);
    imshow(tallImage)
    title('Choose tile in Raw Image')
    
    [rows columns numberOfColorBands] = size(tallImage);

    hold on
    M = size(tallImage,1);
    N = size(tallImage,2);
    a=ceil(M/ivalue); 
    b=ceil(N/jvalue);
    for k = 1:a:M
        x = [1 N]; 
        y = [k k];
        plot(x,y,'Color','black','LineStyle','-');
        set(findobj('Tag','MyGrid'),'Visible','on')
    end
    for k = 1:b:N 
        x = [k k]; 
        y = [1 M];
        plot(x,y,'Color','black','LineStyle','-');
        set(findobj('Tag','MyGrid'),'Visible','on')
    end
    hold off

    ax=gca; 
    hpoint=impoint(ax);
    p=getPosition(hpoint);
    x=uint32(p(1));
    y=uint32(p(2));
    j=ceil(double(x)/double(ceil(tile_size(1))*0.1));
    i=ceil(double(y)/double(ceil(tile_size(2))*0.1));
    if ivalue < 100
       iindex = num2str(100+i);
    else
       iindex = num2str(1000+i);
    end
    iindex = iindex(2:end);
    if jvalue < 100
       jindex = num2str(100+j);
    else
       jindex = num2str(1000+j);
    end
    jindex = jindex(2:end);
    fullfilename = [basename,'_','i',iindex,'j',jindex,'.tif'];

    imdata = imread(fullfilename);

    if roi_count < 1
        [color_markers_temp,image_data,Red,Green,Blue,roi_count] = select_roi(imdata,roi_count,image_count,numberOfColorBands);
    else
        [color_markers_temp,image_data,Red,Green,Blue,roi_count] = select_roi(imdata,roi_count,image_count,numberOfColorBands,color_markers_temp,image_data,Red,Green,Blue);
    end
 
% Construct a questdlg with three options to further continue segmentation or
% testing
choice = questdlg('Do you want to select images again?', ...
'Question', ...
'Yes','No','Yes');
% Handle response
switch choice
case 'Yes'    
    image_count = image_count + 1;
    continue;
case 'No'
    color_markers(1,1) = mean2(color_markers_temp(:,1));
    color_markers(1,2) = mean2(color_markers_temp(:,2));
    color_markers(1,3) = mean2(color_markers_temp(:,3));
    break;
end
end
    
distance_calc(image_count,color_markers,color_markers_temp,image_data,Red,Green,Blue,numberOfColorBands);  
