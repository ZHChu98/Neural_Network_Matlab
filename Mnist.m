classdef Mnist
	properties
    	train;
        test;
    end
   
    methods
        function obj = Mnist(train_images_filepath, train_labels_filepath, ...
                test_images_filepath, test_labels_filepath)
            obj.train = MnistData(train_images_filepath, train_labels_filepath);
            obj.test = MnistData(test_images_filepath, test_labels_filepath);
        end
    end
end