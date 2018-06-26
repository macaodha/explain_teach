import json
import random
import numpy as np


def save_teaching_sequence(teacher, alg_name, op_file_name):
    # saves the teaching sequences so they can be used by the webapp
    results = {}
    num_train = len(teacher.teaching_exs)
    results['num_train'] = num_train

    if 'rand' not in alg_name:
        if 'strict' in alg_name:
            results['im_ids'] = teacher.teaching_exs
            results['display_explain_image'] = [0 for ii in range(num_train)]
        if 'explain' in alg_name:
            results['im_ids'] = teacher.teaching_exs
            results['display_explain_image'] = [1 for ii in range(num_train)]

        with open(op_file_name, 'w') as js:
            json.dump(results, js)


def save_teaching_images(dataset_train, dataset_test, op_file_name, url_root):

    teaching_ims = []
    for ii in range(len(dataset_train['im_files'])):
        im_data = {}
        im_data['image_url'] = url_root + dataset_train['im_files'][ii]
        im_data['explain_url'] = url_root + dataset_train['explain_files'][ii]
        im_data['class_label'] = dataset_train['Y'][ii]
        teaching_ims.append(im_data)

    # puts test images at the end
    for ii in range(len(dataset_test['im_files'])):
        im_data = {}
        im_data['image_url'] = url_root + dataset_test['im_files'][ii]
        im_data['class_label'] = dataset_test['Y'][ii]
        teaching_ims.append(im_data)

        with open(op_file_name, 'w') as js:
            json.dump(teaching_ims, js)


def save_settings(dataset_train, dataset_test, experiment_id, num_random_test_ims, scale, op_file_name):
    settings = {}
    settings['experiment_id'] = experiment_id
    settings['train_indices'] = range(len(dataset_train['im_files']))
    settings['test_indices'] = [len(dataset_train['im_files'])+tt for tt in range(len(dataset_test['im_files']))]
    settings['test_sequence'] = [-1]*num_random_test_ims
    settings['class_names'] = [cc.replace('_', ' ') for cc in dataset_train['class_names']]
    settings['scale'] = scale

    with open(op_file_name, 'w') as js:
        json.dump(settings, js)
