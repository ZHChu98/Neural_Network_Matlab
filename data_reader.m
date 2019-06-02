function [mnist] = data_reader(work_dir)
    train_images_filepath = [work_dir, '/', 'train-images-idx3-ubyte'];
    train_labels_filepath = [work_dir, '/', 'train-labels-idx1-ubyte'];
    test_images_filepath = [work_dir, '/', 't10k-images-idx3-ubyte'];
    test_labels_filepath = [work_dir, '/', 't10k-labels-idx1-ubyte'];
    mnist = Mnist(train_images_filepath, train_labels_filepath, ...
        test_images_filepath, test_labels_filepath);
end