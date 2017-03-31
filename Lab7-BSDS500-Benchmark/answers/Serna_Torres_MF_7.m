%%  Median filter to segmentations 
% validation kmeans 

kVroot = fullfile(LabRoot,'results','kmeans','valR');
matrices =dir([kVroot,'/*.mat']);

for i = 1:numel(matrices)
   name =  matrices(i).name;
   name = name(1:end-4);
  
   A = load(fullfile(LabRoot,'results','kmeans','valR',matrices(i).name));
   im = A.segs;
   segs =cell(1,6);
  for j=1:6
      %Applies a median filter that takes into account only 7 neighbors 
      B = medfilt2(im{1,j},[7 7]);
      idx = find(B==0);
      B(idx) = 1;
      segs{1,j} = B;
  end
   save(fullfile(LabRoot,'results','kmeans','val_MF',name),'segs')
end

%% Evaluation 

imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/val';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/val';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/kmeans/val_MF';
outDir = 'eval/valMF0_kmeans';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)

%% Test kmeans 
kTroot = fullfile(LabRoot,'results','kmeans','test');
matrices =dir([kTroot,'/*.mat']);

for i = 1:numel(matrices)
   name =  matrices(i).name;
   name = name(1:end-4);
  
   A = load(fullfile(LabRoot,'results','kmeans','test',matrices(i).name));
   im = A.segs;
   segs =cell(1,6);
  for j=1:6
       
      B = medfilt2(im{1,j},[7 7]);
      idx = find(B==0);
      B(idx) = 1;
      segs{1,j} = B;
  end
   save(fullfile(LabRoot,'results','kmeans','test_MF',name),'segs')
end

%% Evaluation test kmeans 

imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/test';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/test';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/kmeans/test_MF';
outDir = 'eval/testMF0_kmeans';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)
%% Test GMM 
kTroot = fullfile(LabRoot,'results','GMM','test');
matrices =dir([kTroot,'/*.mat']);

for i = 1:numel(matrices)
   name =  matrices(i).name;
   name = name(1:end-4);
  
   A = load(fullfile(LabRoot,'results','GMM','test',matrices(i).name));
   im = A.segs;
   segs =cell(1,6);
  for j=1:6
       
      B = medfilt2(im{1,j},[7 7]);
      idx = find(B==0);
      B(idx) = 1;
      segs{1,j} = B;
  end
   save(fullfile(LabRoot,'results','GMM','test_MF',name),'segs')
end

%% Evaluation test kmeans 

imgDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/images/test';
gtDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/BSR/BSDS500/data/groundTruth/test';
inDir = '/Users/mariajosesernaayala/Documents/MATLAB/201710/vision/lab7/results/GMM/test_MF';
outDir = 'eval/testMF0_GMM';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)