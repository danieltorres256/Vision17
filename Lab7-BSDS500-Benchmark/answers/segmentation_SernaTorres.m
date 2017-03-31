
function segmentation = segmentation_SernaTorres(rgbimage, clusteringMethod, numberOfClusters)
    
    % Feature space lab+xy
    image2 = rgb2lab(rgbimage);
    x = meshgrid(1:size(image2, 1), 1:size(image2,2));
    y = meshgrid(1:size(image2, 2), 1:size(image2,1));
    image2(:, :, 4) = x';
    image2(:, :, 5) = y;
    
 
    
    if strcmp(clusteringMethod, 'kmeans')  
        % Number of dimensions os the feature space 
        numdim = size(rgbimage,3);
    
    
    % Segmentation in lab  
    imageb = double(rgb2lab(rgbimage));
    nrow2 = size(imageb,1);
    ncol2 = size(imageb,2);
    
    features2 = reshape(imageb,nrow2*ncol2,numdim);
    [cluster_idx2] = kmeans(features2, numberOfClusters,'distance','sqEuclidean', 'Replicates', 3);
    segmentation = reshape(cluster_idx2, nrow2,ncol2);

    elseif strcmp(clusteringMethod, 'gmm')  
    % Number of dimensions os the feature space 
    numdim=size(image2,3);
    
      
    % Segmentation in lab+xy 
    image2 = double(image2);
    nrow = size(image2,1);
    ncol = size(image2,2);
    
    features = reshape(image2,nrow*ncol,numdim);
    try
    jerar_idx = fitgmdist(features, numberOfClusters); 
    catch exception
    jerar_idx = fitgmdist(features, numberOfClusters,'RegularizationValue',0.1);
    end
    
    idx = cluster(jerar_idx, features);
    segmentation = col2im(idx, [1 1], size(image2(:,:,1)));
    

end
  