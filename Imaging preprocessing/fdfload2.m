function out=fdfload2(path)
%Created by Carlos J. Perez Torres. June 2012
%Loads mri files from VnmrJ. Will grab all fdf files in the img folder.
%Output includes all non singleton dimensions organized as follows: read,
%phase, echo, arrayed parameter, slice

%% Get the folder to open from user
if nargin < 1
    path=uigetdir;
end

files=dir(fullfile(path,'*.fdf'));
if isempty(files)
    out=[];
    return
end


%% Open the first file and read the header
fid=fopen(fullfile(path,files(end).name));%%Generates a pointer that identifies the file
num = 0;
done = false;
echoes=0; images=0; slices=0;
machineformat = 'ieee-be'; % Old Unix-based
line = fgetl(fid);
while (~isempty(line) && ~done)
    line = fgetl(fid);
    % disp(line)
    
    if strncmp('int    bigendian', line,16)
        machineformat = 'ieee-le'; % New Linux-based
    end
    
    if strncmp('float  matrix[] = ', line,18)
        [token, rem] = strtok(line,'float  matrix[] = { , };');
        M(1) = str2double(token);
        [token, rem] = strtok(rem,', }');
        M(2) = str2double(token);
        if size(rem,2)>2
            M(3) = str2double(strtok(rem,', };'));
        else
            M(3)= -1; %Marks that scan is 2D
        end
    end %Slice dimensions
    
    if strncmp('int    slices = ', line,16)
        token = strtok(line,'int    slices = { , };');
        slices=str2double(token);
    end %Number of slices
    
    if strncmp('int    echoes = ', line,16)
        token = strtok(line,'int    echoes = { , };');
        echoes=str2double(token);
    end %For a multi-echo sequence
    
    if strncmp('float  array_dim = ', line,19)
        token = strtok(line,'float  array_dim = { , };');
        images=str2double(token);
    end %For when an array is used
    
    if strncmp('float  bits = ', line,14)
        token = strtok(line,'float  bits = { , };');
        bits = str2double(token);
    end
    
    num = num + 1;
    
    if num > 41
        done = true;
    end
end
fclose(fid);

%% Use the header information to read the files and arrange matrix
if M(3)<0
    out=zeros(M(2),M(1),echoes,images,slices);
    for i=1:echoes
        for j=1:images
            for k=1:slices
                fid=fopen(fullfile(path,files((k-1)*images*echoes+(j-1)*echoes+i).name));

                fseek(fid, -M(1)*M(2)*bits/8, 'eof');
              
                out(:,:,i,j,k) = (fread(fid, [M(1), M(2)], 'float32', machineformat))';
                fclose(fid);
            end
        end
    end
    out=squeeze(out);
    if ndims(out)==4
        out=permute(out,[1,2,4,3]);
    elseif ndims(out)==5
        out=permute(out,[1,2,5,3,4]);
    end
else
    out=zeros(M(1),M(2),M(3),echoes,images);
    for i=1:echoes
        for j=1:images
            fid=fopen(fullfile(path,files((j-1)*echoes+i).name));
            
            fseek(fid,-M(1)*M(2)*M(3)*bits/8,'eof');
            
            temp=(fread(fid,M(1)*M(2)*M(3),'float32',machineformat))';
            out(:,:,:,i,j) = flipdim(reshape(temp,[M(1) M(2) M(3)]),1);
            fclose(fid);
        end
    end
    out=squeeze(out);
end



end