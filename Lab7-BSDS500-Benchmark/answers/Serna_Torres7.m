%% Lab 7 BSDS500
LabRoot = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7';
ImRoot = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data';
addpath('/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/bench_fast/benchmarks')

%% Creating folders to save segmentations 

mkdir results
cd results 
mkdir kmeans 
mkdir GMM 

cd kmeans
mkdir train
mkdir trainR
mkdir test
mkdir val
mkdir valR
cd ..
cd GMM 
mkdir train
mkdir trainR
mkdir test
mkdir val
mkdir valR

cd .. 
%% Training kmeans 

% Images directory 
imTRoot = fullfile('images','train');
imValRoot = fullfile('images','val');
dirImTraining = dir([fullfile(ImRoot,imTRoot),'/*.jpg']);
dirImVal = dir([fullfile(ImRoot,imValRoot),'/*.jpg']);

for i = 1:numel(dirImTraining)
   name =  dirImTraining(i).name;
   name = name(1:end-4);
   
   A = imread(fullfile(ImRoot,imTRoot,dirImTraining(i).name));
   seg = cell(1,6); 
   seg{1,1} = segmentation_SernaTorres(A,'kmeans',3); 
   seg{1,2} = segmentation_SernaTorres(A,'kmeans',4); 
   seg{1,3} = segmentation_SernaTorres(A,'kmeans',5);
   seg{1,4} = segmentation_SernaTorres(A,'kmeans',6);
   seg{1,5} = segmentation_SernaTorres(A,'kmeans',7);
   seg{1,6} = segmentation_SernaTorres(A,'kmeans',8);
   
   save(fullfile(LabRoot,'results','kmeans','train',name),'seg')
end

%% Evaluation train kmeans
kTroot = fullfile(LabRoot,'results','kmeans','train');
matrices =dir([kTroot,'/*.mat']);


for i = 1:numel(matrices)
   name =  matrices(i).name;
   name = name(1:end-4);
   
   A = load(fullfile(LabRoot,'results','kmeans','train',matrices(i).name));
   segs = A.seg; 
  
   save(fullfile(LabRoot,'results','kmeans','trainR',name),'segs')
end


imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/train';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/train';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/kmeans/trainR';
outDir = 'eval/train_kmeans';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)


%% Validation kmeans 

for i = 1:numel(dirImVal)
   name =  dirImVal(i).name;
   name = name(1:end-4);
   
   A = imread(fullfile(ImRoot,imValRoot,dirImVal(i).name));
   seg = cell(1,6); 
   seg{1,1} = segmentation_SernaTorres(A,'kmeans',3); 
   seg{1,2} = segmentation_SernaTorres(A,'kmeans',5); 
   seg{1,3} = segmentation_SernaTorres(A,'kmeans',7);
   seg{1,4} = segmentation_SernaTorres(A,'kmeans',9);
   seg{1,5} = segmentation_SernaTorres(A,'kmeans',11);
   seg{1,6} = segmentation_SernaTorres(A,'kmeans',13);
   
   save(fullfile(LabRoot,'results','kmeans','val',name),'seg')
end


%% Evaluation validation kmeans 

kVroot = fullfile(LabRoot,'results','kmeans','val');
matrices =dir([kVroot,'/*.mat']);


for i = 1:numel(matrices)
   name =  matrices(i).name;
   name = name(1:end-4);
   
   A = load(fullfile(LabRoot,'results','kmeans','val',matrices(i).name));
   segs = A.seg; 
  
   save(fullfile(LabRoot,'results','kmeans','valR',name),'segs')
end


imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/val';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/val';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/kmeans/valR';
outDir = 'eval/val_kmeans';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)
%% Testing kmeans 

imTestRoot = fullfile('images','test');
dirImTest = dir([fullfile(ImRoot,imTestRoot),'/*.jpg']);

for i = 1:numel(dirImTest)
   name =  dirImTest(i).name;
   name = name(1:end-4);
   
   A = imread(fullfile(ImRoot,imTestRoot,dirImTest(i).name));
   segs = cell(1,6); 
   segs{1,1} = segmentation_SernaTorres(A,'kmeans',3); 
   segs{1,2} = segmentation_SernaTorres(A,'kmeans',5); 
   segs{1,3} = segmentation_SernaTorres(A,'kmeans',7);
   segs{1,4} = segmentation_SernaTorres(A,'kmeans',9);
   segs{1,5} = segmentation_SernaTorres(A,'kmeans',11);
   segs{1,6} = segmentation_SernaTorres(A,'kmeans',13);
   
   save(fullfile(LabRoot,'results','kmeans','test',name),'segs')
end

%% Evaluation test KMEANS
imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/test';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/test';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/kmeans/test';
outDir = 'eval/test_kmeans';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)


%% Training GMM 

for i = 157:numel(dirImTraining)
   name =  dirImTraining(i).name;
   name = name(1:end-4);
   
   A = imread(fullfile(ImRoot,imTRoot,dirImTraining(i).name));
   seg = cell(1,6); 
   seg{1,1} = segmentation_SernaTorres(A,'gmm',3); 
   seg{1,2} = segmentation_SernaTorres(A,'gmm',5); 
   seg{1,3} = segmentation_SernaTorres(A,'gmm',7);
   seg{1,4} = segmentation_SernaTorres(A,'gmm',9);
   seg{1,5} = segmentation_SernaTorres(A,'gmm',11);
   seg{1,6} = segmentation_SernaTorres(A,'gmm',13);
   
   save(fullfile(LabRoot,'results','GMM','train',name),'seg')
end


%% Evaluation training GMM  

gTroot = fullfile(LabRoot,'results','GMM','train');
matricesGT =dir([gTroot,'/*.mat']);


for i = 1:numel(matricesGT)
   name =  matricesGT(i).name;
   name = name(1:end-4);
   
   A = load(fullfile(LabRoot,'results','GMM','train',matricesGT(i).name));
   segs = A.seg; 
  
   save(fullfile(LabRoot,'results','GMM','trainR',name),'segs')
end


imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/train';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/train';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/kmeans/trainR';
outDir = 'eval/train_GMM';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)

%% validation GMM 

for i = 80:numel(dirImVal)
   name =  dirImVal(i).name;
   name = name(1:end-4);
   
   A = imread(fullfile(ImRoot,imValRoot,dirImVal(i).name));
   seg = cell(1,6); 
   seg{1,1} = segmentation_SernaTorres(A,'gmm',3); 
   seg{1,2} = segmentation_SernaTorres(A,'gmm',5); 
   seg{1,3} = segmentation_SernaTorres(A,'gmm',7);
   seg{1,4} = segmentation_SernaTorres(A,'gmm',9);
   seg{1,5} = segmentation_SernaTorres(A,'gmm',11);
   seg{1,6} = segmentation_SernaTorres(A,'gmm',13);
   
   save(fullfile(LabRoot,'results','GMM','val',name),'seg')
end
%% Evaluation validation GMM 
gVroot = fullfile(LabRoot,'results','GMM','val');
matricesGV =dir([gVroot,'/*.mat']);


for i = 1:numel(matricesGV)
   name =  matricesGV(i).name;
   name = name(1:end-4);
   
   A = load(fullfile(LabRoot,'results','GMM','val',matricesGV(i).name));
   segs = A.seg; 
  
   save(fullfile(LabRoot,'results','GMM','valR',name),'segs')
end


imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/val';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/val';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/GMM/valR';
outDir = 'eval/EVAL_GMM';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)
%% Testing GMM 

imTestRoot = fullfile('images','test');
dirImTest = dir([fullfile(ImRoot,imTestRoot),'/*.jpg']);

for i = 1:numel(dirImTest)
   name =  dirImTest(i).name;
   name = name(1:end-4);
   
   A = imread(fullfile(ImRoot,imTestRoot,dirImTest(i).name));
   segs = cell(1,6); 
   segs{1,1} = segmentation_SernaTorres(A,'gmm',3); 
   segs{1,2} = segmentation_SernaTorres(A,'gmm',5); 
   segs{1,3} = segmentation_SernaTorres(A,'gmm',7);
   segs{1,4} = segmentation_SernaTorres(A,'gmm',9);
   segs{1,5} = segmentation_SernaTorres(A,'gmm',11);
   segs{1,6} = segmentation_SernaTorres(A,'gmm',13);
   
   save(fullfile(LabRoot,'results','GMM','test',name),'segs')
end


%% Evaluation test GMM 

imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/test';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/test';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/GMM/test';
outDir = 'eval/test_GMM';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)