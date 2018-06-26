import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import datetime as dt
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import confusion_matrix
sns.set_style("whitegrid")


def get_time_diff(user_time):
    times = [dt.datetime.strptime(tt, '%H:%M:%S') for tt in user_time]
    time_diff = []
    for tt in range(len(times)-1):
        diff = (times[tt+1] - times[tt]).seconds
        time_diff.append(diff)
    time_diff = np.hstack((0, time_diff))
    return time_diff


# select which daatset to plot
exp_name = 'oct'
#exp_name = 'butterflies_crop'
#exp_name = 'chinese_chars'
#exp_name = 'chinese_chars_crowd'


plt.close('all')
save_data = False
remove_bottom = False
rm_fraction = 0.2

col_p = sns.color_palette()
col_p[3], col_p[2] = col_p[2], col_p[3]
majorLocator = MultipleLocator(5)
minorLocator = MultipleLocator(1)
majorFormatter = FormatStrFormatter('%d')


if not save_data:
    print '***\nNot saving plots\n***\n'

if remove_bottom:
    print '***\nRemoving worst workers\n***\n'
else:
    print '***\nUsing all data\n***\n'


base_dir = 'experiments/' + exp_name + '/'
results_file = base_dir + 'results.json'
settings_file = base_dir + 'settings.json'
op_dir = base_dir + '/plots/'
if (not os.path.isdir(op_dir)) and save_data:
    os.makedirs(op_dir)

# load data
with open(settings_file) as f:
    settings = json.load(f)
num_classes = len(settings['class_names'])

with open(results_file) as f:
    user_data = json.load(f)

strats = [uu['strategy'] for uu in user_data]
un_strats = np.unique(strats)
ip_strats_names = ['random', 'random_feedback', 'strict_1vall', 'explain_1vall']
ip_strats_names_full = ['RAND_IM', 'RAND_EXP', 'STRICT', 'EXPLAIN']
op_strat_names = [ip_strats_names_full[ip_strats_names.index(ss)] for ss in un_strats]

if (np.unique(un_strats) == np.unique(strats)).mean() != 1.0:
    print '\n*****Warning - missing strat\n*******'

print '\n', len(user_data), 'users completed the task'
print 'strategies\t', un_strats
print 'classes   \t', settings['class_names']


train_inds = np.where(np.asarray(user_data[0]['is_train'])==1)[0]
test_inds = np.where(np.asarray(user_data[0]['is_train'])==0)[0]

# load data
train_acc = {}
test_acc = {}
test_pred = {}
test_gt = {}
scores_all = {}
user_times = {}
print '\n'.ljust(20), ' train'.ljust(10), ' test'.ljust(10), 'med tst'.ljust(10), 'num people'
for fid, ss in enumerate(un_strats):
    resp = np.asarray([uu['response'] for uu in user_data if uu['strategy'] == ss])
    gt = np.asarray([uu['gt_label'] for uu in user_data if uu['strategy'] == ss])
    scores = np.asarray([uu['score'] for uu in user_data if uu['strategy'] == ss])
    tm = np.asarray([uu['time'] for uu in user_data if uu['strategy'] == ss])

    if remove_bottom:
        # remove bottom X%
        keep_inds = np.argsort(scores)[int(len(scores)*rm_fraction):]
    else:
        keep_inds = np.arange(scores.shape[0])

    scores = scores[keep_inds].copy()
    resp = resp[keep_inds, :].copy()
    gt = gt[keep_inds, :].copy()
    tm = tm[keep_inds, :].copy()

    train_acc[ss] = (resp[:, train_inds]==gt[:, train_inds]).mean(0)*100
    test_acc[ss] = (resp[:, test_inds]==gt[:, test_inds]).mean(0)*100
    scores_all[ss] = scores
    user_times[ss] = tm
    test_pred[ss] = resp[:, test_inds]
    test_gt[ss] = gt[:, test_inds]

    print ss.ljust(20), str(round(np.mean(train_acc[ss]),2)).ljust(10), str(round(np.mean(test_acc[ss]),2)).ljust(10), str(round(np.median(scores),2)).ljust(10), len(scores)


# hist of test acc
fig = plt.figure(0, figsize=(7, 6))
fig.suptitle(exp_name + ' - Test Accuracy', fontsize=14)
for fid, ss in enumerate(un_strats):
    plt.subplot(2, 2, fid+1)
    if ss == 'explain_1vall':
        plt.hist(scores_all[ss], bins=10, range=(0,100), color=col_p[3])
    else:
        plt.hist(scores_all[ss], bins=10, range=(0,100), color=col_p[0])

    plt.xlim(0, 100)
    plt.title(op_strat_names[fid])
plt.tight_layout(rect=[0, 0, 1, 0.95])

if save_data:
    plt.savefig(op_dir + '0.png')
    plt.savefig(op_dir + '0.pdf')


# test acc boxplot
fig = plt.figure(1)
fig.suptitle(exp_name + ' - Test Accuracy', fontsize=14)
plt.boxplot([scores_all[ss].tolist() for ss in un_strats], labels=un_strats)
plt.ylim([0, 100])

if save_data:
    plt.savefig(op_dir + '1.png')
    plt.savefig(op_dir + '1.pdf')


# train accuracy over time
fig = plt.figure(2)
fig.suptitle(exp_name + ' - Train Accuracy', fontsize=14)
for fid, ss in enumerate(un_strats):
    plt.plot(np.arange(len(train_acc[ss]))+1, train_acc[ss], label=op_strat_names[fid], color=col_p[fid])
plt.legend()
plt.xlabel('Training Image')
plt.ylabel('Average Accuracy')

plt.gca().xaxis.set_major_locator(majorLocator)
plt.gca().xaxis.set_major_formatter(majorFormatter)
plt.gca().xaxis.set_minor_locator(minorLocator)

plt.ylim([0, 100])
plt.xlim([1, train_inds.shape[0]])
plt.show()

if save_data:
    plt.savefig(op_dir + '2.png')
    plt.savefig(op_dir + '2.pdf')


# test accuracy over time
fig = plt.figure(3)
fig.suptitle(exp_name + ' - Test Accuracy', fontsize=14)
for fid, ss in enumerate(un_strats):
    plt.plot(np.arange(len(test_acc[ss]))+1, test_acc[ss], label=op_strat_names[fid], color=col_p[fid])
plt.legend()
plt.ylim([0, 100])
plt.xlim([1, test_inds.shape[0]])
plt.show()

if save_data:
    plt.savefig(op_dir + '3.png')
    plt.savefig(op_dir + '3.pdf')


# average time against accuracy
fig = plt.figure(4)
fig.suptitle(exp_name + ' - Test Timings', fontsize=14)
for fid, ss in enumerate(un_strats):
    time_diff = [np.mean(get_time_diff(uut)[test_inds]) for uut in user_times[ss]]  # mean of workers
    plt.plot(time_diff, scores_all[ss], '.', label=op_strat_names[fid])
plt.legend()
plt.ylim([0, 102])
plt.xlim(xmin=0)
plt.xlabel('Mean time (seconds)')
plt.ylabel('Test accuracy')

if save_data:
    plt.savefig(op_dir + '4.png')
    plt.savefig(op_dir + '4.pdf')


# average time by strategy
fig = plt.figure(5)
fig.suptitle(exp_name + ' - Test Timings', fontsize=14)
mean_times = []
for fid, ss in enumerate(un_strats):
    time_diff = [np.mean(get_time_diff(uut)[test_inds]) for uut in user_times[ss]]  # mean of workers
    mean_times.append(time_diff)

plt.boxplot(mean_times, labels=un_strats)
plt.ylabel('Test Time (seconds per image)')

if save_data:
    plt.savefig(op_dir + '5.png')
    plt.savefig(op_dir + '5.pdf')


# confusion matrices
cms = []
for fid, ss in enumerate(un_strats):
    cm = confusion_matrix(test_gt[ss].ravel(), test_pred[ss].ravel())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cms.append(cm*100)

if len(un_strats) == 4:
    fig, axes = plt.subplots(nrows=2, ncols=2, num=6)
else:
    fig, axes = plt.subplots(nrows=1, ncols=2, num=6)

fig.suptitle(exp_name + ' - Average Class Confusion', fontsize=14)
for fid, ss in enumerate(un_strats):
#for fid, ax in enumerate(axes.flat):
    ax = axes.flat[fid]
    im = ax.imshow(cms[fid], cmap='plasma', vmin=0, vmax=100.0)
    ax.set_yticks(np.arange(num_classes))
    ax.grid('off')
    ax.set_title(op_strat_names[fid])
plt.tight_layout(rect=[0, 0, 1, 0.95])
cax = fig.add_axes([0.9, 0.1, 0.03, 0.75])
fig.colorbar(im, cax=cax)

if save_data:
    plt.savefig(op_dir + '6.png')
    plt.savefig(op_dir + '6.pdf')


plt.show()