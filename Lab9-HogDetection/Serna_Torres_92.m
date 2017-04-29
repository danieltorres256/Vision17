%% Face detection

%% Directories 
LabRoot = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9';
%LabRoot ='/datos1/vision/SernaTorres/Lab9/';
DirPatches = fullfile(LabRoot,'lab9Detection','TrainCrops', 'TrainCrops');
imFolders = dir(DirPatches);

VLFEATROOT = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/vlfeat';
run(fullfile(VLFEATROOT,'toolbox','vl_setup'));
%%
for k = length(imFolders):-1:1
    %remueve las carpetas que no hacen parte del directorio.
    if ~imFolders(k).isdir
        imFolders(k) = [ ];
        continue
    end
    fname = imFolders(k).name;
    %remueve los dor archivos que estan siempre dentro de las carpetas.
    %'.','..'
    if fname(1) == '.'
        imFolders(k) = [ ];
    end
end

%% 
hogCellSize = 8;
trainHog = {} ;
trainBoxPatches = {};
for i = 1:numel(imFolders)
    
    imNames = dir([fullfile(DirPatches,imFolders(i).name),'/*.jpg']);
    for j = 1:numel(imNames)
        trainBoxPatches{j,i} = imresize(imread(fullfile(DirPatches,imFolders(i).name,imNames(j).name)),[100 100]);
    end
end


idx = find(~cellfun(@isempty,trainBoxPatches));

for i = 1: numel(idx)
   trainBoxPatchesDEF{i,1} = trainBoxPatches{idx(i)}; 
end


for i = 1:numel(trainBoxPatchesDEF)
    
   trainHog{i} = vl_hog(im2single(trainBoxPatchesDEF{i}), hogCellSize);
      
end
%%
trainHog = cat(4, trainHog{:});
w = mean(trainHog, 4); 

%% Multiple scales
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(minScale,maxScale,numOctaveSubdivisions*(maxScale-minScale+1)) ;

%% Negatives

Negativos = load('Negatives.mat');
Negativos = Negativos.defNeg;

% HOG for negative samples 
trainHogNeg = {};
for i = 1:numel(Negativos)
    
   trainHogNeg{i,1} = vl_hog(im2single(imresize(Negativos{i},[100 100])), hogCellSize);
      
end

%% 
trainHogNeg = cat(4, trainHogNeg{:});
wNeg = mean(trainHogNeg, 4); 

%%
pos = trainHog;
neg = trainHogNeg;

numPos = size(pos,4) ;
numNeg = size(neg,4) ;
C = 10 ;
lambda = 1 / (C * (numPos + numNeg)) ;

x = cat(4, pos, neg) ;
x = reshape(x, [], numPos + numNeg) ;

y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;
% Learn the SVM using an SVM solver
w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;


%% Detection 

dirImPr= '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/WIDER_val/images';
foldersIm = dir(dirImPr);

%%
for k = length(foldersIm):-1:1
    %remueve las carpetas que no hacen parte del directorio.
    if ~foldersIm(k).isdir
        foldersIm(k) = [ ];
        continue
    end
    fname = foldersIm(k).name;
    %remueve los dor archivos que estan siempre dentro de las carpetas.
    %'.','..'
    if fname(1) == '.'
        foldersIm(k) = [ ];
    end
end
%%

for i=1:numel(foldersIm)
   
   imagenes = dir([fullfile(dirImPr,foldersIm(i).name),'/*.jpg']); 
   for j=1:numel(imagenes)
       
   end
end