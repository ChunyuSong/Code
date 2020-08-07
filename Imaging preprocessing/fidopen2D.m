function [Images,ud]=fidopen2D(zerofill,path)
%Created by Carlos J. Perez Torres. February 2014
%Modified in December 2014 for multi-echo
%Opens a 2D FID file

if nargin < 1
    zerofill=1;
end


%% Get the folder to open from user
if nargin < 2
    path=uigetdir;
end

fidfile=dir(fullfile(path,'fid'));
if isempty(fidfile)
    error('No FID file found')
end
procfile=dir(fullfile(path,'procpar'));
if isempty(procfile)
    error('No procpar file found')
end

%% Read the files
ud=readprocpar(fullfile(path,'procpar'));
[fiddata,HDR]=readfid(fullfile(path,'fid'));
% HDR
% size(fiddata)

%% Reconstruct the FID
if ud.ni==1 && HDR.traces_per_block/ud.ns~=ud.nv %There is a weird case for diffusion scans with no NI
    temp=reshape(complex(fiddata(8:2:end,:),fiddata(9:2:end,:)),HDR.elements_per_trace/2,2*ud.ns,HDR.traces_per_block/ud.ns/2,HDR.blocks/ud.ni,ud.ni);
    temp=permute(temp,[1,5,3,2,4]);
    temp=reshape(temp,HDR.elements_per_trace/2,HDR.traces_per_block/ud.ns*ud.ni/2,ud.ns*2,HDR.blocks/ud.ni);
    kspace=temp(:,:,1:2:2*ud.ns,:);
%     fprintf('kspace dimensions...');
%     size(kspace)
else
    kspace=reshape(complex(fiddata(8:2:end,:),fiddata(9:2:end,:)),HDR.elements_per_trace/2,ud.ne,ud.ns,HDR.traces_per_block/(ud.ns*ud.ne),HDR.blocks/ud.ni,ud.ni);
%     fprintf('kspace dimensions1...');
%     size(kspace)
    
    kspace=permute(kspace,[1,6,3,2,4,5]);
%     fprintf('kspace dimensions2...');
%     size(kspace)
    
    kspace=squeeze(reshape(kspace,HDR.elements_per_trace/2,HDR.traces_per_block/(ud.ns*ud.ne)*ud.ni,ud.ns,ud.ne,HDR.blocks/ud.ni));
    fprintf('kspace dimensions3...');
    size(kspace)
end

%Flipping whole image along all dimensions. Origin definition is probably flipped?
kspace=flipdim(flipdim(kspace,1),2);
if zerofill>1
    if ud.ne==1
        new_kspace=zeros(size(kspace,1)*zerofill,size(kspace,2)*zerofill,size(kspace,3));
        diff=size(new_kspace)-size(kspace);
        new_kspace(diff(1)/2+1:end-diff(1)/2,diff(2)/2+1:end-diff(2)/2,:)=kspace;
        kspace=new_kspace;
    else
        new_kspace=zeros(size(kspace,1)*zerofill,size(kspace,2)*zerofill,size(kspace,3),size(kspace,4));
        diff=size(new_kspace)-size(kspace);
        new_kspace(diff(1)/2+1:end-diff(1)/2,diff(2)/2+1:end-diff(2)/2,:,:)=kspace;
        kspace=new_kspace;
    end
end

Images=zeros(zerofill*ud.np/2,zerofill*ud.nv,ud.ns,ud.ne,ud.arraydim/ud.ni);
[~ , sliceOrder]=sort(ud.pss);
for i=1:ud.arraydim/ud.ni
    for sliceno=1:ud.ns
        for echoes=1:ud.ne
            picture=ifftshift(ifftshift(squeeze(kspace(:,:,sliceOrder(sliceno),echoes,i)),1),2);
%             imshow(abs(picture),[])
            Images(:,:,sliceno,echoes,i)=abs(fftshift(fftshift(fft(fft(picture,[],1),[],2),1),2));
        end
    end
end


%Correct phase and readout shifts
% roshift=fix(ud.pro/ud.lro*ud.np/2);
% if roshift>0
%     temp=Images;
%     Images(1:ud.np/2-roshift,:,:)=temp(roshift+1:ud.np/2,:,:);
%     Images(ud.np/2-roshift+1:ud.np/2,:,:)=temp(1:roshift,:,:);
% elseif roshift<0
%     roshift=-roshift;
%     temp=Images;
%     Images(1:roshift,:,:)=temp(ud.np/2-roshift+1:ud.np/2,:,:);
%     Images(roshift+1:ud.np/2,:,:)=temp(1:ud.np/2-roshift,:,:);
% 
% end

% phaseshift=fix(ud.ppe/ud.lpe*ud.nv);
% if phaseshift>0
%     temp=Images;
%     Images(:,1:phaseshift,:)=temp(:,ud.nv-phaseshift+1:ud.nv,:);
%     Images(:,phaseshift+1:ud.nv,:)=temp(:,1:ud.nv-phaseshift,:);
% elseif phaseshift<0
%     phaseshift=-phaseshift;
%     temp=Images;
%     Images(:,1:ud.nv-phaseshift,:)=temp(:,phaseshift+1:ud.nv,:);
%     Images(:,ud.nv-phaseshift+1:ud.nv,:)=temp(:,1:phaseshift,:);
% end


% Reshape to fit Matlab conventions
% Images=permute(Images,[2,1,3]);
% Images=flipdim(Images,3);

% a=size(Images);
% implay(real(reshape(Images,[a(1),a(2),1,a(3)]))./max(abs(Images(:))));
end

%% ---- Function for loading procpar
function procpar = readprocpar(procfile)
procpar=[];
%% Open file
fid=fopen(procfile,'r','ieee-be');

%% Read whole file into cell array
C = textscan(fid,'%s','delimiter','\n','BufSize',1024*1024);
procpar_str=C{1};


%% Close file
fclose(fid);

%% Parse procpar file lines
%try
  nonlabelchars='123456789';
  field_chars = '_1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
  nonfield_chars = char(setdiff(0:256,double(field_chars)));
  for ii=1:length(procpar_str)
    if any(procpar_str{ii}(1)==nonlabelchars)
      %% Read info
      str=deblank(procpar_str{ii});
      ind=find(str==' ');
      if isempty(ind)
        ind=0;
      end
      str=str(ind(1)+1:end);
      
      if str(1)=='"'
        ind2=find(str=='"');
        for kk=1:2:(length(ind2))
          procpar.(label){end+1} = strrep(str(ind2(kk):ind2(kk+1)),'"','');
        end
      else
        procpar.(label) = str2num(str);
      end
    elseif procpar_str{ii}(1)=='"' % Read string from line
      str=deblank(procpar_str{ii});
      procpar.(label){end+1} = strrep(str,'"','');
    elseif procpar_str{ii}(1)=='0' % Empty line, end of block
      continue
    else
      %% Read label
      ind=find(procpar_str{ii}==' ');
      label=procpar_str{ii}(1:ind-1);
      
      %% Make sure that the characters in the label are compatible with 
      %% Matlab structure fields. If not, replace them with underscore...
      ind2=ismember(label,nonfield_chars);
      if any(ind2)
        label(ind2)='_';
      end
      procpar.(label)={};
    end
  end
  
  %% Parse array parameter. If more than one parameters are arrayed, they
  %  are separated with commas. If parameters are jointly arrayed they
  %  are enclosed in brackets.
  if isfield(procpar,'array') && ~isempty(procpar.array{1})
    str = procpar.array{1};
    
	% Add Matlab cell characters
	str = strrep(strrep(str,'(','{'),')','}');
	str = ['{',str,'}'];
	
	% Add string characters around words
	str=regexprep(str,'(\w+)(,|\})','''$1''$2');
	
	% Evaluate to formulate s cell
	procpar.array = eval(str);
	
  end
  
end


%% ---- Function for loading fids
function [fiddata, HDR] = readfid(fidfile)

fid = fopen(fidfile,'r','ieee-be');

% Global header
HDR.blocks             = fread(fid,1,'int32');
HDR.traces_per_block   = fread(fid,1,'int32');
HDR.elements_per_trace = fread(fid,1,'int32');
HDR.bytes_per_element  = fread(fid,1,'int32');
HDR.bytes_per_trace    = fread(fid,1,'int32');
HDR.bytes_per_block    = fread(fid,1,'int32');
HDR.version            = fread(fid,1,'int16');
HDR.status             = fread(fid,1,'int16');
HDR.block_headers      = fread(fid,1,'int32');

statusbits = dec2bin(HDR.status,8);
% check if data is stored as int32 (vnmr) or float (vnmrj)
if str2num(statusbits(5))==1
    HDR.precision = 'float';
else
    HDR.precision = 'int32';
end

fiddata=reshape(fread(fid,7*HDR.blocks+HDR.blocks*HDR.traces_per_block*HDR.elements_per_trace*2,HDR.precision),[],HDR.blocks);
% fprintf('Fid data size...');
% size(fiddata)

fclose(fid);
end