classdef MnistData
	properties
        images;
        labels;
        size;
        pos;
    end
   
	methods
        function obj = MnistData(images_filepath, labels_filepath, one_hot)
            fp = fopen(images_filepath, 'r');
            assert(fp ~= -1, ['Could not open ', images_filepath]);
            
            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2051, ['Bad magic number in', images_filepath]);
            
            n_images = fread(fp, 1, 'int32', 0, 'ieee-be');
            n_rows = fread(fp, 1, 'int32', 0, 'ieee-be');
            n_cols = fread(fp, 1, 'int32', 0, 'ieee-be');
            
            obj.images = fread(fp, inf, 'unsigned char');
            obj.images = reshape(obj.images, [n_cols, n_rows, n_images]);
            obj.images = permute(obj.images, [2, 1, 3]); % different with python
            obj.images = obj.images / 256.0;
            
            fclose(fp);
            
            fp = fopen(labels_filepath, 'r');
            assert(fp ~= -1, ['Could not open ', labels_filepath]);
            
            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2049, ['Bad magic number in ', labels_filepath, '']);
            
            n_labels = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(n_labels == n_images, 'Mismatch in label count');
            obj.size = n_images;
            
            obj.labels = fread(fp, inf, 'unsigned char');
            
            fclose(fp);
            
            if one_hot == true
                tmp = zeros(obj.size, 10);
                for i = 1:obj.size
                    tmp(i, obj.labels(i) + 1) = 1;
                end
            end
            obj.labels = tmp;
            
            obj.pos = 0;
        end 
	end
end