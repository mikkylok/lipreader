import os
import pickle


def combine_pickle(dataset_split):
    pickle_dir = os.path.join('../pickle', dataset_split)
    label_dirs = [label for label in os.listdir(pickle_dir) if label != '.DS_Store']
    total_data = []
    for label in label_dirs:
        folder_path = os.path.join(pickle_dir, label)
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file != '.DS_Store']
        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                total_data.append(data)
    output_file = dataset_split + '.pkl'
    with open(output_file, 'wb') as file:
        pickle.dump(total_data, file)


def check_data(pkl_path):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
        return data


if __name__ == '__main__':
    combine_pickle('test')
    # check_data('train.pkl')
