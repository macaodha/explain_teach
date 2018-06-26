# Helper functions that deal with file loading

from __future__ import print_function
import numpy as np
import os
import json


def load_ims(im_file):
    # load images and class names
    print('\nloading images')
    with open(im_file, 'r') as fs:
        images = json.load(fs)

    for ii, im in enumerate(images):
        if 'explain_url' not in im.keys():
            im['explain_url'] = im['image_url']

    for ii, im in enumerate(images):
        print(ii, os.path.basename(im['image_url']))

    print(len(images), 'images loaded')
    return images


def load_settings(settings_file):
    print('\nloading settings')
    with open(settings_file, 'r') as fs:
        settings = json.load(fs)
    return settings['class_names'], settings['train_indices'], settings['test_indices'], settings['test_sequence'], settings['experiment_id'], settings['scale']


def load_strats(strat_files, test_sequence):
    print('\nloading strategy files')
    strats = {}
    for ii, sf in enumerate(strat_files):
        strat_name = os.path.basename(sf)[:-6]
        print(ii, strat_name)
        with open(sf, 'r') as fp:
            strat_data = json.load(fp)
        res = {}

        res['num_train'] = strat_data['num_train']
        res['num_test'] = len(test_sequence)

        if 'random' in strat_name:
            res['test_sequence'] = list(test_sequence)
            res['display_explain_image'] = strat_data['display_explain_image']
        else:
            res['image_id'] = strat_data['im_ids'] + list(test_sequence)
            res['display_explain_image'] = strat_data['display_explain_image'] + [0]*res['num_test']
            res['is_train'] = [1]*res['num_train'] + [0]*res['num_test']
        strats[strat_name] = res

    print(len(strats), 'strategies loaded\n')
    return strats


