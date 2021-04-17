from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES, dataset_names_for_archive
from utils.utils import read_all_datasets
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import run_length_xps
from utils.utils import generate_results_csv

from utils import utils
import numpy as np
import sys
import sklearn
import os


def prepare_data():
    X_train = datasets_dict[dataset_name][0]
    Y_train = datasets_dict[dataset_name][1]
    X_test = datasets_dict[dataset_name][2]
    Y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((Y_train, Y_test), axis=0)))

    # make the min to zero of labels
    Y_train, Y_test = transform_labels(Y_train, Y_test)

    # save orignal y because later we will use binary
    y_true = Y_test.astype(np.int64)
    y_true_train = Y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((Y_train, Y_test), axis=0).reshape(-1, 1))
    Y_train = enc.transform(Y_train.reshape(-1, 1)).toarray()
    Y_test = enc.transform(Y_test.reshape(-1, 1)).toarray()

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, Y_train, X_test, Y_test, y_true, nb_classes, y_true_train, enc


def fit_classifier(classifier_type):
    X_train, Y_train, X_test, Y_test, Y_true, nb_classes, y_true_train, enc = prepare_data()

    input_shape = X_train.shape[1:]

    classifier = create_classifier(classifier_type, input_shape, nb_classes,
                                   output_directory)

    classifier.fit(X_train, Y_train, X_test, Y_test, Y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape,
                                  nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                              build=build)


def get_xp_val(xp):
    if xp == 'batch_size':
        xp_arr = [16, 32, 128]
    elif xp == 'use_bottleneck':
        xp_arr = [False]
    elif xp == 'use_residual':
        xp_arr = [False]
    elif xp == 'nb_filters':
        xp_arr = [16, 64]
    elif xp == 'depth':
        xp_arr = [3, 9]
    elif xp == 'kernel_size':
        xp_arr = [8, 64]
    else:
        raise Exception('wrong argument')
    return xp_arr


############################################### main
if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    root_dir = '.'
    data_dir = "../data"
    xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
           'kernel_size', 'batch_size']

    if sys.argv[1] == 'InceptionTime':
        # run nb_iter_ iterations of Inception on the whole TSC archive
        classifier_name = 'inception'
        archive_name = ARCHIVE_NAMES[0]
        nb_iter_ = 5

        datasets_dict = read_all_datasets(data_dir, archive_name)

        for iter in range(nb_iter_):
            print('\t\titer', iter)

            trr = ''
            if iter != 0:
                trr = '_itr_' + str(iter)

            tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

            for dataset_name in dataset_names_for_archive[archive_name]:
                print('\t\t\tdataset_name: ', dataset_name)

                output_directory = tmp_output_directory + dataset_name + '/'

                temp_output_directory = create_directory(output_directory)

                if temp_output_directory is None:
                    print('Already_done', tmp_output_directory, dataset_name)
                    continue

                fit_classifier(classifier_name)

                print('\t\t\t\tDONE')

                # the creation of this directory means
                create_directory(output_directory + '/DONE')

        # run the ensembling of these iterations of Inception
        classifier_name = 'nne'
        print(f"{classifier_name}".center(80, "-"))
        datasets_dict = read_all_datasets(data_dir, archive_name)

        tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

        for dataset_name in dataset_names_for_archive[archive_name]:
            print('\t\t\tdataset_name: ', dataset_name)

            # x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

            output_directory = tmp_output_directory + dataset_name + '/'

            fit_classifier(classifier_name)

            print('\t\t\t\tDONE')

    elif sys.argv[1] == 'InceptionTime_xp':
        # this part is for running inception with the different hyperparameters
        # listed in the paper, on the whole TSC archive
        archive_name = 'UCRArchive_2018'
        classifier_name = 'inception'
        max_iterations = 5

        datasets_dict = read_all_datasets(data_dir, archive_name)

        for xp in xps:

            xp_arr = get_xp_val(xp)

            print('xp', xp)

            for xp_val in xp_arr:
                print('\txp_val', xp_val)

                kwargs = {xp: xp_val}

                for iter in range(max_iterations):

                    trr = ''
                    if iter != 0:
                        trr = '_itr_' + str(iter)
                    print('\t\titer', iter)

                    for dataset_name in dataset_names_for_archive[archive_name]:

                        output_directory = root_dir + '/results/' + classifier_name + '/' + '/' + xp + '/' + '/' + str(
                            xp_val) + '/' + archive_name + trr + '/' + dataset_name + '/'

                        print('\t\t\tdataset_name', dataset_name)
                        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                        # check if data is too big for this gpu
                        size_data = x_train.shape[0] * x_train.shape[1]

                        temp_output_directory = create_directory(output_directory)

                        if temp_output_directory is None:
                            print('\t\t\t\t', 'Already_done')
                            continue

                        input_shape = x_train.shape[1:]

                        from classifiers import inception

                        classifier = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                    verbose=False, build=True, **kwargs)

                        classifier.fit(x_train, y_train, x_test, y_test, y_true)

                        # the creation of this directory means
                        create_directory(output_directory + '/DONE')

                        print('\t\t\t\t', 'DONE')

        # we now need to ensemble each iteration of inception (aka InceptionTime)
        archive_name = ARCHIVE_NAMES[0]
        classifier_name = 'nne'

        datasets_dict = read_all_datasets(data_dir, archive_name)

        tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

        for xp in xps:
            xp_arr = get_xp_val(xp)
            for xp_val in xp_arr:

                clf_name = 'inception/' + xp + '/' + str(xp_val)

                for dataset_name in dataset_names_for_archive[archive_name]:
                    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                    output_directory = tmp_output_directory + dataset_name + '/'

                    from classifiers import nne

                    classifier = nne.Classifier_NNE(output_directory, x_train.shape[1:],
                                                    nb_classes, clf_name=clf_name)

                    classifier.fit(x_train, y_train, x_test, y_test, y_true)

    elif sys.argv[1] == 'run_length_xps':
        # this is to generate the archive for the length experiments
        run_length_xps(root_dir)

    elif sys.argv[1] == 'generate_results_csv':
        clfs = []
        itr = '-0-1-2-3-4-'
        inceptionTime = 'nne/inception'
        # add InceptionTime: an ensemble of 5 Inception networks
        clfs.append(inceptionTime + itr)
        # add InceptionTime for each hyperparameter study
        for xp in xps:
            xp_arr = get_xp_val(xp)
            for xp_val in xp_arr:
                clfs.append(inceptionTime + '/' + xp + '/' + str(xp_val) + itr)
        df = generate_results_csv('results.csv', root_dir, clfs)
        print(df)