clear; close all;

%% Configuration 
% NOTE: you can modify this part
train_set = 'train';
scale = 3;
hr_size = 48;
stride = 48;

%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts
disp(sprintf('%10s: %s', 'Train set', train_set));
disp(sprintf('%10s: %d', 'Scale', scale));

scale_dir = strcat(int2str(scale), 'x');

% example: 
% read_path = '../data/train'
% save_path = '../preprocessed_data_video/train/3x/'
read_path = fullfile('../original_data', train_set) 
save_path = fullfile('../preprocessed_data', train_set, scale_dir);

if exist(save_path, 'dir') ~= 7
	mkdir(save_path)
end

% count variable to order the data
count = 0;

data = zeros(hr_size, hr_size, 1, 1);
label = zeros(hr_size, hr_size, 1, 1);


filepaths = [];
filepaths = [filepaths; dir(fullfile(read_path, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(read_path, '*.jpg'))];

for i = 1 : length(filepaths)
	disp(sprintf('Processing image: %d', i));
	for i_scale = 0.6: 0.2: 1.0 % augment scale
		for i_rot = 0: 90: 270
			for i_flip = 1:3
				image = imread(fullfile(read_path, filepaths(i).name));
				
				image = imrotate(image, i_rot);
				if (i_flip == 1 || i_flip == 2) % flip vert and horz
					image = flipdim(image, i_flip);
				end
				image = imresize(image, i_scale, 'bicubic');

				if size(image, 3) == 3
					image_ycbcr = rgb2ycbcr(image);
					image_y = image_ycbcr(:, :, 1);
				end

				hr_im = im2double(image_y);
				hr_im = modcrop(hr_im, scale);
				[hei, wid] = size(hr_im);
				lr_im = imresize(hr_im,1/scale,'bicubic');
				lr_im = imresize(lr_im ,[hei, wid],'bicubic');

				for h = 1 : stride : hei - hr_size + 1
					for w = 1 : stride : wid - hr_size + 1

						hr_sub_im = hr_im(h:hr_size+h-1, w:hr_size+w-1);
						lr_sub_im = lr_im(h:hr_size+h-1, w:hr_size+w-1);

						count = count + 1;

						data(:, :, 1, count) = lr_sub_im;
						label(:, :, 1, count) = hr_sub_im;
						%imwrite(hr_sub_im, strcat('../tmp/',int2str(count),'.png'))
						%imwrite(lr_sub_im, strcat('../tmp/',int2str(count),'l.png'))
					end
				end
			end
		end
	end
end


% writing to HDF5
chunksz = 32;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(fullfile(save_path, 'dataset.h5'), batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(fullfile(save_path, 'dataset.h5'));

