import matplotlib.pyplot as plt
import numpy as np
import offline_teachers as teach
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


def compute_interpretability(explains, Y, Y_pred, interp_sigma):
    # WARNING might choose examples with no explanations that will be biased to being picked

    # want to be high if its a good explanation
    ent = np.zeros(Y.shape[0])
    for ii in range(Y.shape[0]):
        explain_pred = explains[ii, :, :, Y[ii]].copy()
        explain_pred -= explain_pred.min()
        #ent[ii] = entropy(explain_pred.ravel())
        explain_pred /= explain_pred.max()
        aa = explain_pred.ravel() + 0.0000001
        ent[ii] = -(np.log(aa)*aa).mean()

    # if the predctions from CNN don't match the GT we should discourage showing this example
    for ii in range(Y.shape[0]):
        if Y_pred[ii] != Y[ii]:
            ent[ii] = ent.max()

    # remove the class mean entropy - to prevent bias towards some classes
    min_ent = ent.min()
    for cc in np.unique(Y):
        inds = np.where(Y==cc)[0]
        mu = ent[inds].mean()
        ent[inds] -= mu

    # put in the range [0,1], 0 is easiest, 1 is hardest
    ent -= ent.min()
    ent /= ent.max()

    # low entropy means discounting more
    ent = 1.0 / (1.0 + np.exp(-interp_sigma*(ent + 0.0000001)))
    return ent


def compute_likelihood_one_vs_all(hyps, X, Y, alpha):
    likelihoods = []
    if np.unique(Y).shape[0] == 2:
        # binary
        ll = compute_likelihood(hyps, X, Y, alpha, True)
        likelihoods.append(ll)
    else:
        # multi class
        for cc in np.unique(Y):
            Y_bin = np.zeros(Y.shape[0]).astype(np.int)
            Y_bin[np.where(Y==cc)[0]] = 1
            ll = compute_likelihood(hyps, X, Y_bin, alpha, True)
            likelihoods.append(ll)
    return likelihoods


def compute_likelihood(hyps, X, Y, alpha, one_v_all=False):
    # compute P(y|h,x) - size HxN
    # is set to one where h(x) = y i.e. correct guess
    likelihood = np.ones((len(hyps), X.shape[0]))
    likelihood_opp = np.ones((len(hyps), X.shape[0]))

    for hh in range(len(hyps)):
        if one_v_all:
            # assumes that hyps[hh] is a D dim vector
            prob_agree, pred_class = teach.user_model_binary(hyps[hh], X, Y, alpha)
        else:
            # assumes that hyps[hh] is a CxD dim maxtrix
            prob_agree, pred_class = teach.user_model(hyps[hh], X, Y, alpha)
        inds = np.where(pred_class != Y)[0]
        likelihood[hh, inds] = prob_agree[inds]

    return likelihood


def compute_density(X, Y, sigma, per_class=True):
    # compute the density of the datapoints
    dist = squareform(pdist(X)**2)
    if per_class:
        dens = np.zeros((X.shape[0]))
        for cc in np.unique(Y):
            inds = np.where(Y==cc)[0]
            dens[inds] = dist[inds, :][:, inds].mean(1)
    else:
        dens = dist.mean(1)
    dens = 1.0 / (1.0  + np.exp(-sigma*dens))

    return dens


def plot_2D_data(X, Y, alpha, hyps, random_exs, post, title_txt, fig_id, one_v_all, best_ind):
    plt.figure(fig_id)
    plt.title(title_txt)

    # plot hyper-planes
    l_weight_range = (0.5, 10)
    delta = 1.0
    xx = np.linspace(X[:,0].min()-delta, X[:,0].max()+delta)
    for hh in range(len(hyps)):
        if one_v_all:
            ww = hyps[hh]
        else:
            ww = hyps[hh][1,:]  # for binary this is positive class


        if ww.shape[0] == 3:
            # with intercept i.e. ww[2]
            m = -ww[0] / ww[1]
            yy = m * xx - (ww[2]) / ww[1]
        else:
            # no intercept
            yy = (-ww[0] / ww[1])*xx

        plt.plot(xx,yy, 'g')

    # plot datapoints and text labels
    for ii, ll in enumerate(random_exs):
        plt.text(X[ll, 0]+0.1, X[ll, 1]+0.1, ii)
        plt.plot(X[ll, 0], X[ll, 1], 'yo')

    cols = ['r.', 'b.', 'c.', 'm.', 'k.']
    for ii, yy in enumerate(np.unique(Y)):
        plt.plot(X[Y==yy,0], X[Y==yy,1], cols[ii])

    plt.axis('equal')
    delta = 0.5
    plt.axis([X[:,0].min()-delta,X[:,0].max()+delta, X[:,1].min()-delta, X[:,1].max()+delta])
    plt.show()


def plot_2D_data_hyper(X, Y, alpha, hyps, random_exs, post, title_txt, fig_id, one_v_all, best_ind):
    # TODO this doesnt work for 1 v all need to plot hyper plans separately per class

    # this plots the data points X, the labels Y, along with the different
    # hypotheses hyp and their associated posterior weights
    # currently only works for binary classes
    # also best to use rand hypotheses for 2D datasets
    plt.figure(fig_id)
    plt.title(title_txt)

    # plot hyper-planes
    l_weight_range = (0.5, 10)
    delta = 1.0
    xx = np.linspace(X[:,0].min()-delta, X[:,0].max()+delta)
    for hh in range(len(hyps)):
        if one_v_all:
            print 'WARNING this is only implemented for binary'
            ww = hyps[hh]
        else:
            ww = hyps[hh][1,:]  # for binary this is positive class

        if ww.shape[0] == 3:
            # with intercept i.e. ww[2]
            m = -ww[0] / ww[1]
            yy = m * xx - (ww[2]) / ww[1]
        else:
            # no intercept
            yy = (-ww[0] / ww[1])*xx

        lw = post[hh]*(l_weight_range[1]-l_weight_range[0]) + l_weight_range[0]
        if hh == best_ind:
            plt.plot(xx,yy, 'r', linewidth=lw)  # optimal hypothesis
        else:
            plt.plot(xx,yy, 'g', linewidth=lw)  # regular hypothesis

    # plot datapoints and text labels
    for ii, ll in enumerate(random_exs):
        plt.text(X[ll, 0]+0.1, X[ll, 1]+0.1, ii)
        plt.plot(X[ll, 0], X[ll, 1], 'yo')

    cols = ['r.', 'b.', 'k.', 'c.', 'm.']
    for ii, yy in enumerate(np.unique(Y)):
        plt.plot(X[Y==yy,0], X[Y==yy,1], cols[ii])

    plt.axis('equal')
    delta = 0.5
    plt.axis([X[:,0].min()-delta,X[:,0].max()+delta, X[:,1].min()-delta, X[:,1].max()+delta])
    plt.show()
