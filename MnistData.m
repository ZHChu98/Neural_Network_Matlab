classdef MnistData
    properties
        images;
        labels;
        size;
    end

    methods
        function obj = MnistData(images_filepath, labels_filepath)
            fp = fopen(images_filepath, 'r');
            assert(fp ~= -1, ['Could not open ', images_filepath]);
            
            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2051, ['Bad magic number in', images_filepath]);
            
            n_images = fread(fp, 1, 'int32', 0, 'ieee-be');
            n_rows = fread(fp, 1, 'int32', 0, 'ieee-be');
            n_cols = fread(fp, 1, 'int32', 0, 'ieee-be');
            
            obj.images = fread(fp, inf, 'unsigned char');
            obj.images = reshape(obj.images, [n_cols, n_rows, 1, n_images]);
            obj.images = permute(obj.images, [2, 1, 3, 4]); % different with python
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
        end
        
        function [batch_images, batch_labels, new_pos] = next_batch(obj, batch_size, pos)
            if pos+batch_size-1 > obj.size
                pos = 1;
            end
            batch_images = obj.images(:, :, 1, pos:pos+batch_size-1);
            batch_labels = obj.labels(pos:pos+batch_size-1);
            new_pos = pos + batch_size;
        end
	end
end