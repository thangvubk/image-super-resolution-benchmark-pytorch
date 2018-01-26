clear; close all;

%% Configuration
% NOTE: you can modify this part
test_set = 'set5'; %()
scale = 3;  % (2, 3, 4)

%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts
disp(sprintf('%10s: %s', 'Test set', test_set));
disp(sprintf('%10s: %d', 'Scale', scale));

scale_dir = strcat(int2str(scale), 'x');

% example
% read_path = '../data/test/set5/'
% save_path = '../preprocessed_data//test/set5/3x/'
read_path = fullfile('../original_data', 'test', test_set);
save_path = fullfile('../preprocessed_data', 'test', test_set, scale_dir);
high_res_save_path = fullfile(save_path, 'high_res')
low_res_save_path = fullfile(save_path, 'low_res')

if exist(high_res_save_path, 'dir') ~= 7
	mkdir(high_res_save_path)
end

if exist(low_res_save_path, 'dir') ~= 7
	mkdir(low_res_save_path)
end

filepaths = dir(fullfile(read_path, '*.bmp'));

for i = 1 : length(filepaths)
	image = imread(fullfile(read_path, filepaths(i).name));
	if size(image, 3) == 3
		image_ycbcr = rgb2ycbcr(image);
		image_y = image_ycbcr(:, :, 1);
	end
	hr_im = im2double(image_y);
	hr_im = modcrop(hr_im, scale);
	[hei, wid] = size(hr_im);
	lr_im = imresize(hr_im,1/scale,'bicubic');
	lr_im = imresize(lr_im ,[hei, wid],'bicubic');
	
	imwrite(hr_im, fullfile(high_res_save_path, filepaths(i).name))
	imwrite(lr_im, fullfile(low_res_save_path, filepaths(i).name))
end

