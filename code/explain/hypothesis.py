import numpy as np
from sklearn.cluster import KMeans
import random
from sklearn import svm
import offline_teachers as teach
import itertools


def compute_hyps_error_one_vs_all(hyps, X, Y, alpha):
    # compute err(h, h*) - list of length C with vectors of H

    err_hyps = []
    if np.unique(Y).shape[0] == 2:
        # if only two classes don't need to do both
        err_hyps.append(compute_hyps_error(hyps, X, Y, alpha, True))
    else:
        # multi class
        for cc in np.unique(Y):
            Y_bin = np.zeros(Y.shape[0]).astype(np.int)
            Y_bin[np.where(Y==cc)[0]] = 1
            err = compute_hyps_error(hyps, X, Y_bin, alpha, True)
            err_hyps.append(err)

    return err_hyps


def compute_hyps_error(hyps, X, Y, alpha, one_v_all=False):
    # compute err(h, h*) - vector of length H
    err_hyp = np.zeros(len(hyps))
    for hh in range(len(hyps)):
        if one_v_all:
            _, pred_class = teach.user_model_binary(hyps[hh], X, Y, alpha)
        else:
            _, pred_class = teach.user_model(hyps[hh], X, Y, alpha)
        err_hyp[hh] = (Y != pred_class).sum() / float(Y.shape[0])

    return err_hyp


def cluster_hyps(X, Y, num_hyps, alpha, clf):
    # this is for the multi class hypothesis case - number of combinations explodes

    # generate hypotheses by clustering data
    num_classes = np.unique(Y).shape[0]

    # all possible combinations of two classes
    class_combs = []
    for ii in range(np.minimum(num_classes-1,2)):
        class_combs.extend(itertools.combinations(range(num_classes), ii + 1))

    classifiers = []
    for cc in class_combs:
        inds = []
        for ii in cc:
            inds.append(np.where(Y==ii)[0])
        inds = np.hstack(inds)

        # fit SVM
        tmp_labels = np.zeros(Y.shape[0]).astype(np.int)
        tmp_labels[inds] = 1
        clf.fit(X, tmp_labels)
        classifiers.append(clf.coef_[0,:].copy())
    classifiers = np.vstack(classifiers)

    # cant use all possible combinations, just use a subset
    all_combs = list(itertools.permutations(range(len(classifiers)), num_classes))
    subset_combs = random.sample(range(len(all_combs)), np.minimum(num_hyps,len(all_combs)))
    print len(all_combs), 'possible combinations of PW classifers'

    # copy hypothesis
    hyps = []
    for ss in subset_combs:
        inds = all_combs[ss]
        hyps.append(classifiers[inds, :].copy())

    # add teacher i.e. best hypothesis - trained on all data
    clf.fit(X, Y)
    hyps.append(clf.coef_.copy())

    return hyps


def cluster_hyps_one_v_all(X, Y, alpha, clf):
    # generate hypotheses by clustering data for 1 versus all

    hyps = []
    num_classes = np.unique(Y).shape[0]
    clusters_per_class = 2

    # 1) sub classes against the rest
    cinds = 0
    Y_hal = np.zeros(Y.shape[0]).astype(np.int)
    for cc in np.unique(Y):
        inds = np.where(Y==cc)[0]
        kmeans = KMeans(n_clusters=clusters_per_class).fit(X[inds, :])
        Y_hal[inds] = kmeans.labels_.copy()+cinds
        cinds += clusters_per_class

    for cc in np.unique(Y_hal):
        inds = np.where(Y_hal==cc)[0]
        tmp_labels = np.zeros(Y_hal.shape[0]).astype(np.int)
        tmp_labels[inds] = 1
        clf.fit(X, tmp_labels)
        hyps.append(clf.coef_[0,:].copy())

    # 2) each class against the rest - GT
    for cc in np.unique(Y):
        inds = np.where(Y==cc)[0]
        tmp_labels = np.zeros(Y.shape[0]).astype(np.int)
        tmp_labels[inds] = 1
        clf.fit(X, tmp_labels)
        hyps.append(clf.coef_[0,:].copy())

    # 3) pairs of classes against the rest
    combs = list(itertools.combinations(range(num_classes), 2))
    for cc in combs:
        inds = []
        for cur_class in cc:
            inds.append(np.where(Y==cur_class))
        tmp_labels = np.zeros(Y.shape[0]).astype(np.int)
        tmp_labels[np.hstack(inds)] = 1
        clf.fit(X, tmp_labels)
        hyps.append(clf.coef_[0,:].copy())

    return hyps


def sparse_hyps(X, Y, num_hyps, one_v_all, clf, fit_gt=False):
    # sparse hypotheses with small number of -1 or 1 entries
    num_non_zero = 2
    num_classes = np.unique(Y).shape[0]
    hyps = []
    for hh in range(num_hyps):
        if one_v_all or (num_classes == 2):
            w = np.zeros((X.shape[1]))
            inds = random.sample(range(X.shape[1]), num_non_zero)
            w[inds] = np.random.choice([-1,1], num_non_zero)
        else:
            w = np.zeros((num_classes, X.shape[1]))
            for cc in range(num_classes):
                inds = random.sample(range(X.shape[1]), num_non_zero)
                w[cc, inds] = np.random.choice([-1,1], num_non_zero)
        hyps.append(w)

    # add GT
    if fit_gt:
        clf.fit(X, Y)
        if one_v_all:
            for cc in range(num_classes):
                hyps.append(clf.coef_[cc, :].copy())
        else:
            hyps.append(clf.coef_.copy())

    return hyps


def random_hyps(X, Y, num_hyps, alpha, one_v_all, clf, fit_gt=False):
    # generate random set of hypotheses
    num_classes = np.unique(Y).shape[0]
    hyps = []
    for hh in range(num_hyps):
        if one_v_all:
            hyp = np.random.randn(X.shape[1])
        elif num_classes == 2:
            hh = np.random.randn(X.shape[1])
            hyp = np.vstack((-hh, hh))
        else:
            hyp = np.random.randn(num_classes, X.shape[1])
        hyps.append(hyp)

    # add GT
    if fit_gt:
        clf.fit(X, Y)
        if one_v_all:
            for cc in range(num_classes):
                hyps.append(clf.coef_[cc, :].copy())
        elif num_classes == 2:
            hyps.append(np.vstack((-clf.coef_.copy(), clf.coef_.copy())))
        else:
            hyps.append(clf.coef_.copy())

    return hyps


def generate_hyps(dataset, alpha, num_hyps, hyp_type, one_v_all):
    # generates the hypothesis space
    # if one_v_all is True we create D dim hypothesis otherwise we do CxD
    X = dataset['X']
    Y = dataset['Y']
    num_classes = np.unique(Y).shape[0]
    clf = svm.LinearSVC(fit_intercept=False, penalty='l1', loss='squared_hinge', dual=False)

    # if only 2D we will add negative versions of hyps later for visualization
    # <= 3 as we might have bias term
    if X.shape[1] <= 3 and num_classes == 2:
        print '2D dataset -> generating less hypotheses'
        num_hyps /= 2

    if hyp_type == 'cluster':
        if one_v_all:
            hyps = cluster_hyps_one_v_all(X, Y, alpha, clf)
        else:
            hyps = cluster_hyps(X, Y, num_hyps, alpha, clf)

    if hyp_type == 'cluster_rand':
        if one_v_all:
            hyps = cluster_hyps_one_v_all(X, Y, alpha, clf)
        else:
            hyps = cluster_hyps(X, Y, num_hyps, alpha, clf)

        if num_hyps - len(hyps) > 0:
            hyps.extend(random_hyps(X, Y, num_hyps - len(hyps), alpha, one_v_all, clf, False))

    elif hyp_type == 'rand':
        hyps = random_hyps(X, Y, num_hyps, alpha, one_v_all, clf, True)
    elif hyp_type == 'sparse':
        hyps = sparse_hyps(X, Y, num_hyps, one_v_all, clf, True)

    # if 2D data add negative versions of hypothesis - makes visualization easier
    if X.shape[1] <= 3 and num_classes == 2:
        hyps_opposite = []
        for hh in range(len(hyps)):
            hyps_opposite.append(hyps[hh].copy()*-1)
        hyps.extend(hyps_opposite)

    # create prior
    prior_h = np.ones(len(hyps)) / float(len(hyps))

    return hyps, prior_h
