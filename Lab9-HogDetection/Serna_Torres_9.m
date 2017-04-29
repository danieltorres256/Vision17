clc, clear all, close all 
% Laboratory 9 Mariajosé Serna - Henry Daniel Torres 

%% Finding negatives 
LabRoot = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9';
dirIm= '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/WIDER_val/images';
folders = dir(dirIm);

dirAnot = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/eval_tools_mod/ground_truth';
anot = load(fullfile(dirAnot,'wider_easy_val.mat'));
foldAnotList = anot.face_bbx_list;
fileNames = anot.file_list;

% Creating folder for negative images 
mkdir /Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9 negatives 
 %
for k = length(folders):-1:1
    %remueve las carpetas que no hacen parte del directorio.
    if ~folders(k).isdir
        folders(k) = [ ];
        continue
    end
    fname = folders(k).name;
    %remueve los dor archivos que estan siempre dentro de las carpetas.
    %'.','..'
    if fname(1) == '.'
        folders(k) = [ ];
    end
end

%
rng(0,'twister')

% Limits for random numbers 
% Size range for imcrops sizes
upLimS = 164; downLimS = 80; 

for i = 1:numel(folders)
    % aqui estoy en un folder
    dirCompl = fullfile(dirIm,folders(i).name);
    % list of bounding boxes of the images of the current folder
    AnotactFolder = foldAnotList{i};
    filesActNames = fileNames{i};
    for j = 1:numel(AnotactFolder)
        % aqui recorro las imagenes del folder
        AnotImAct = AnotactFolder{j};
        imNames = filesActNames{j};
        % numeros enteros para las anotaciones
        AnotImAct = ceil(AnotImAct);
        completeName =strcat(imNames,'.jpg');
        image = imread(fullfile(dirCompl,completeName));
        
        idxN = find(AnotImAct<1);
        if numel(idxN) ~= 0
            break
            
        else
            for k = 1:size(AnotImAct,1)
                
                
                coorA = AnotImAct(k,1); coorB = AnotImAct(k,2);
                sizeA = AnotImAct(k,3); sizeB = AnotImAct(k,4);
                sumA = coorA+sizeA;
                sumB = coorB+sizeB;
                image(coorB:sumB,coorA:sumA,:) = 0;
                
                
            end
        end
        
        tamano = ceil((upLimS-downLimS).*rand(1,10) + downLimS);
        coorx = ceil(((size(image,1)-tamano)-1).*rand(1,10) + 1);
        coory = ceil(((size(image,2)-tamano)-1).*rand(1,10) + 1);
        
        box = [coorx' coory' tamano' tamano'];
        
        for p = 1:10
            negativos{p,j,i} = imcrop(image,box(p,:));  
        end
    end
end

%% 
idx = find(~cellfun(@isempty,negativos));

for i = 1: numel(idx)
   defNeg{i,1} = negativos{idx(i)}; 
end
%%
name = 'Negativos';
save(['/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/negatives/' name '.mat'],'defNeg')