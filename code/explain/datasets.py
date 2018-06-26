from sklearn import datasets
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import offline_teachers as teach
import utils as ut


def load_datasets(dataset_name, dataset_dir, do_pca, pca_dims, add_bias, remove_mean, density_sigma, interp_sigma):
    print dataset_name

    im_files = None
    explain_files = None
    class_names = None
    explain_interp = None  # for the explanation 1.0 means easy to interpret and 0.0 means hard

    if dataset_name == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        Y = iris.target
    elif dataset_name == 'wine':
        wine = datasets.load_wine()
        X = wine.data
        Y = wine.target
    elif dataset_name == 'breast_cancer':
        bc = datasets.load_breast_cancer()
        X = bc.data
        Y = bc.target
    elif dataset_name == '2d_outlier':
        num_exs = 100
        sig = 0.005
        pt = 0.3
        cls1 = np.random.multivariate_normal([pt, pt], [[sig, 0],[0,sig]], int(num_exs*0.8))
        cls2 = np.random.multivariate_normal([-pt, -pt], [[sig, 0],[0,sig]], int(num_exs*0.8))
        # add "noise"
        cls1n = np.random.multivariate_normal([pt, pt], [[sig*10, 0],[0,sig*10]], int(num_exs*0.2))
        cls2n = np.random.multivariate_normal([-pt, -pt], [[sig*10, 0],[0,sig*10]], int(num_exs*0.2))
        X = np.vstack((cls1, cls1n, cls2, cls2n))
        Y = np.ones(X.shape[0]).astype(np.int)
        Y[:int(num_exs*0.8)+int(num_exs*0.2)] = 0
    elif dataset_name == '3blobs':
        num_exs = 80
        cls1 = np.random.multivariate_normal([1.0, -1.0], [[0.12, 0],[0,0.12]], num_exs)
        cls2 = np.random.multivariate_normal([-1.0, -1.0], [[0.12, 0],[0,0.12]], num_exs)
        cls3 = np.random.multivariate_normal([-1.0, 1.0], [[0.12, 0],[0,0.12]], num_exs)
        X = np.vstack((cls1,cls2, cls3))
        Y = np.ones(X.shape[0]).astype(np.int)
        Y[:num_exs] = 0
    elif dataset_name == 'blobs_2_class':
        X, Y = make_blobs(n_samples=200, centers=2, random_state=0)
    elif dataset_name == 'blobs_3_class':
        X, Y = make_blobs(n_samples=300, centers=3, random_state=0)
    else:
        X, Y, im_files, explain_files, class_names, explain_interp = load_data(dataset_dir, dataset_name, interp_sigma)

    if im_files is None:
        im_files = np.asarray(['']*X.shape[0])
    if explain_files is None:
        explain_files = np.asarray(['']*X.shape[0])
    if class_names is None:
        class_names = np.asarray(['']*np.unique(Y).shape[0])
    if explain_interp is None:
        explain_interp = np.ones(X.shape[0])

    # standardize
    if remove_mean:
        X = X - X.mean(0)
        X = X / X.std(0)

    # do PCA
    if do_pca and X.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit(X)
        X = pca.transform(X)
        X = X - X.mean(0)
        X = X / X.std(0)

    # add 1 for bias (intercept) term
    if add_bias:
        X = np.hstack((X, np.ones(X.shape[0])[..., np.newaxis]))

    # balance datasets - same number of examples per class
    X, Y, im_files, explain_files, explain_interp = balance_data(X, Y, im_files, explain_files, explain_interp)

    # train test split
    dataset_train, dataset_test = make_train_test_split(X, Y, im_files, explain_files, class_names, explain_interp)

    # density of points
    dataset_train['X_density'] = ut.compute_density(dataset_train['X'], dataset_train['Y'], density_sigma, True)

    print 'train split'
    print dataset_train['X'].shape[0], 'instances'
    print dataset_train['X'].shape[1], 'features'
    print np.unique(dataset_train['Y']).shape[0], 'classes'

    return dataset_train, dataset_test


def load_data(dataset_dir, dataset_name, interp_sigma):
    data = np.load(dataset_dir + dataset_name + '.npz')
    X = data['X']
    Y = data['Y']
    im_files = data['im_files']
    explain_files = data['explain_files']
    class_names = data['class_names'].tolist()

    # compute interpretability
    if 'interp' not in data.keys():
        # does not exist so set them all the same
        explain_interp = np.ones(X.shape[0])
    elif len(data['interp'].shape) == 1:
        # already computed
        explain_interp = data['interp']
        explain_interp = 1.0 / (1.0 + np.exp(-interp_sigma*(explain_interp+0.0000001)))
    else:
        # not computed, generate it from explanation images
        print 'computing interpretability'
        explain_interp = ut.compute_interpretability(data['interp'], data['Y'], data['Y_pred'], interp_sigma)

    return X, Y, im_files, explain_files, class_names, explain_interp


def make_train_test_split(X, Y, im_files, explain_files, class_names, explain_interp):
    # split_data = [X_train, X_test, Y_trains, ...]
    split_data = train_test_split(X, Y, im_files, explain_files, explain_interp, test_size=0.2, random_state=0)

    datasets = []
    for dd in range(2):
        dataset = {}
        dataset['X'] = split_data[dd+0]
        dataset['Y'] = split_data[dd+2]
        dataset['im_files'] = split_data[dd+4]
        dataset['explain_files'] = split_data[dd+6]
        dataset['explain_interp'] = split_data[dd+8]
        dataset['class_names'] = class_names
        datasets.append(dataset)

    return datasets[0], datasets[1]


def balance_data(X, Y, im_files, explain_files, explain_interp):
    # ensure there is an equal number of examples per class

    # shuffle
    X,Y,im_files,explain_files,explain_interp = resample(X,Y,im_files,explain_files,explain_interp,replace=False,random_state=0)
    min_cnt = X.shape[0]
    for cc in np.unique(Y):
        if (Y==cc).sum() < min_cnt:
            min_cnt = (Y==cc).sum()

    inds = []
    for cc in np.unique(Y):
        inds.extend(np.where(Y==cc)[0][:min_cnt])

    X = X[inds, :]
    Y = Y[inds]
    im_files = im_files[inds]
    explain_files = explain_files[inds]
    explain_interp = explain_interp[inds]

    return X, Y, im_files, explain_files, explain_interp


def remove_exs(dataset, hyps, err_hyp, alpha, split_name, one_v_all):
    # only keep examples that we can predict with the best hypothesis
    if one_v_all:
        if np.unique(dataset['Y'].shape[0]) == 2:
            # binary
            optimal_index = np.argmin(err_hyp[0])
            _, pred_class = teach.user_model_binary(hyps[optimal_index], dataset['X'], dataset['Y'], alpha)
            inds = np.where(dataset['Y'] == pred_class)[0]
        else:
            # multi class
            correctly_predicted = np.zeros(dataset['Y'].shape[0])
            for cc in range(len(err_hyp)):
                optimal_index = np.argmin(err_hyp[cc])
                Y_bin = np.zeros(dataset['Y'].shape[0]).astype(np.int)
                Y_bin[np.where(dataset['Y']==cc)[0]] = 1
                _, pred_class = teach.user_model_binary(hyps[optimal_index], dataset['X'], Y_bin, alpha)
                correctly_predicted[np.where(Y_bin == pred_class)[0]] += 1
            inds = np.where(correctly_predicted == len(err_hyp))[0]
    else:
        optimal_index = np.argmin(err_hyp)
        _, pred_class = teach.user_model(hyps[optimal_index], dataset['X'], dataset['Y'], alpha)
        inds = np.where(dataset['Y'] == pred_class)[0]
    print dataset['X'].shape[0] - inds.shape[0], split_name, 'examples removed'

    # remove the examples
    dataset['X'] = dataset['X'][inds, :]
    dataset['Y'] = dataset['Y'][inds]
    dataset['im_files'] = dataset['im_files'][inds]
    dataset['explain_files'] = dataset['explain_files'][inds]
    dataset['explain_interp'] = dataset['explain_interp'][inds]
    cls_un, cls_cnt = np.unique(dataset['Y'], return_counts=True)
    if 'X_density' in dataset.keys():
        dataset['X_density'] = dataset['X_density'][inds]

    print '\n', split_name
    for cc in range(len(cls_cnt)):
        print cls_un[cc], dataset['class_names'][cls_un[cc]].ljust(30), '\t', cls_cnt[cc]

    return dataset

