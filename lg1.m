clear all;
close all;
%read image
A = imread('img_1.jpg');
% showimage
subplot(2,3,1), imshow(A), title('original image');
%rgb to gray converstion
A_gray = rgb2gray(A);

subplot(2,3,2), imshow(A_gray), title('gray image');
subplot(2,3,3), imhist(A_gray), title('gray histogram graph');
%histogram equalization
ylim('auto')
A_hist = histeq(A_gray, 256);
subplot(2,3,4),imshow(A_hist), title('histogram equalization');
subplot(2,3,5), imhist(A_hist),title('histogram');
ylim('auto')
% IMAGE RESTORATION
A_noise = imnoise(A_hist, 'salt & pepper', 0.05);
figure, subplot(1,2,1), imshow(A_noise),title('Noisy image');
 
A_fit=medfilt2(A_noise,[5,5], 'symmetric');
subplot(1,2,2), imshow(A_fit),title('filterd image image');

 %edge detection
 A_sobel = edge(A_fit, 'sobel');
 figure, subplot(1,2,1), imshow(A_sobel),title('edged image');
 
 %convert to binary
 A_gray = im2double(A_gray);
 %Gray image converted to binary
A_binary = imbinarize(A_fit,0.7);
subplot(1,2,2), imshow(A_binary), title('binary image');
size(A_gray)
se = strel('disk',12);
A_erode= imerode(A_gray, se);
A_outline = A_gray - A_erode;
imshowpair(A_gray, A_outline, 'montage');
%image segmentation
%using morphological gradient
g = fspecial('sobel'); 
h = sqrt(imfilter(A_gray,g,'replicate').^2 +imfilter(A_gray,g','replicate').^2);
figure, imshow(h)
L = watershed(h);
wr = L ==0;
figure,subplot(1,3,1), imshow(wr);
h2 = imclose(imopen(h, ones(3,3)),ones(3,3));
L2 = watershed(h2);

wr2 = L2 ==0;
subplot(1,3,2), imshow(wr2);
e2 = A_binary;
e2(wr2) = 255;%thresholding
subplot(1,3,3), imshow(e2);
% extract feature
 
%convert B to a label matrix
e2(wr2) = bwlabel(e2(wr2));
A_region = regionprops(e2(wr2),'Area','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation','ConvexArea','FilledArea','EulerNumber','EquivDiameter','Solidity','Extent','Perimeter');


glcm = graycoprops(e2(wr2));
    feat1 = [glcm.Contrast;glcm.Correlation;glcm.Energy;glcm.Homogeneity];
    e = entropy(e2(wr2));