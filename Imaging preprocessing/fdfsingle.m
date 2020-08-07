function out=fdfsingle(target)
%Created by Carlos J. Perez Torres. October,2014
%Loads a single fdf file

%% Get the folder to open from user
if nargin<1
    [file,path]=uigetfile('.fdf', 'Select the fdf file');
    target=fullfile(path,file);
end

%% Open the first file and read the header
fid=fopen(target);%%Generates a pointer that identifies the file
num = 0;
done = false;
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
      
    if strncmp('float  bits =', line,13)
        token = strtok(line,'float  bits = { , };');
        bits = str2double(token);
    end
    
    num = num + 1;
    
    if num > 41
        done = true;
    end
end

%% Use the header information to read the file and arrange matrix

fseek(fid, -M(1)*M(2)*bits/8, 'eof');

out = (fread(fid, [M(1), M(2)], 'float32', machineformat))';
fclose(fid);
            




end