%% Detection 

% Directories 
LabRoot = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9';
DirPatches = fullfile(LabRoot,'lab9Detection','TrainCrops', 'TrainCrops');
imFolders = dir(DirPatches);

% Run VL_Feat library 
VLFEATROOT = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/vlfeat';
run(fullfile(VLFEATROOT,'toolbox','vl_setup'));


% Training cofiguration
targetClass = 1 ;
numHardNegativeMiningIterations = 5 ;
schedule = [1 2 5 5 5] ;

% Scale space configuration
hogCellSize = 8 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;

% Removes folders that do not belong to the patches directory 
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

% loading negatives
negatives = load('Negatives.mat');
negatives = negatives.defNeg;
for i=1:numel(negatives)
trainImages{i} = i ;
end

hogCellSize = 8;
trainHog = {} ;
trainBoxPatches = {};
trainBoxLabels = [] ;
trainBoxes = [] ;
trainBoxImages = {} ;

% Charges train images 
for i = 1:numel(imFolders)
    
    imNames = dir([fullfile(DirPatches,imFolders(i).name),'/*.jpg']);
    for j = 1:numel(imNames)
        trainBoxPatches{j,i} = imresize(imread(fullfile(DirPatches,imFolders(i).name,imNames(j).name)),[100 100]);
    end
end

% Ignores empty cells 
idx = find(~cellfun(@isempty,trainBoxPatches));

for i = 1: numel(idx)
   trainBoxPatchesDEF{i} = trainBoxPatches{idx(i)}; 
   trainBoxLabels(i) = 1 ;
   trainBoxes(:,i) = [0.5 ; 0.5 ; 100.5 ; 100.5] ;
end

% HOG representation for train imcrops 
trainBoxHog = {};
for i = 1:numel(trainBoxPatchesDEF)
    
   trainBoxHog{i} = vl_hog(im2single(trainBoxPatchesDEF{i}), hogCellSize);
      
end

%%
trainBoxHog = cat(4, trainBoxHog{:});
modelWidth = size(trainBoxHog,2) ;
modelHeight = size(trainBoxHog,1) ;

%% Train with hard negative mining 


% Initial positive and negative data
pos = trainBoxHog(:,:,:,ismember(trainBoxLabels,targetClass)) ;
neg = zeros(size(pos,1),size(pos,2),size(pos,3),0) ;

%%
for t=1:numHardNegativeMiningIterations
  numPos = size(pos,4) ;
  numNeg = size(neg,4) ;
  C = 1 ;
  lambda = 1 / (C * (numPos + numNeg)) ;
  
  fprintf('Hard negative mining iteration %d: pos %d, neg %d\n', ...
    t, numPos, numNeg) ;
    
  % Train an SVM model (see Step 2.2)
  x = cat(4, pos, neg) ;
  x = reshape(x, [], numPos + numNeg) ;
  y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;
  w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;
  w = single(reshape(w, modelHeight, modelWidth, [])) ;

  % Plot model
  figure(2) ; clf ;
  imagesc(vl_hog('render', w)) ;
  colormap gray ;
  axis equal ;
  title('SVM HOG model') ;
  
  % Evaluate on training data and mine hard negatives
  figure(3) ;  
  [matches, moreNeg] = ...
    evaluateModel(...
    vl_colsubset(trainImages', schedule(t), 'beginning'), ...
    trainBoxes, trainBoxImages, ...
    w, hogCellSize, scales) ;
  
  % Add negatives
  neg = cat(4, neg, moreNeg) ;
  
  % Remove negative duplicates
  z = reshape(neg, [], size(neg,4)) ;
  [~,keep] = unique(z','stable','rows') ;
  neg = neg(:,:,:,keep) ;  
end

%% Detections 

dirEval = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab9/WIDER_val/images';
foldersEval = dir(dirEval);

[detections, scores] = detect(im, w, hogCellSize, scales) ;
keep = boxsuppress(detections, scores, 0.25) ;
detections = detections(:, keep(1:10)) ;
scores = scores(keep(1:10)) ;

% Plot top detection
figure(3) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detections, 'g', 'linewidth', 2, ...
  'label', arrayfun(@(x)sprintf('%.2f',x),scores,'uniformoutput',0)) ;
title('Multiple detections') ;