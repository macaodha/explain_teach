from __future__ import print_function
from flask import Flask, Response, session, redirect, url_for, request, render_template
import uuid
import random
import numpy as np
import glob
import os
import utils as ut
import datetime
from pymongo import MongoClient
from bson.json_util import dumps
import config

application = Flask(__name__)
application.secret_key = config.SECRET_KEY

# database - hosted on mlab
if config.MONGO_DB_STR is not '':
    client = MongoClient(config.MONGO_DB_STR)
    db = client.get_default_database()
else:
    db = MongoClient().database

# load image data
images_file = 'data/teaching_images.json'
images = ut.load_ims(images_file)

# load other settings
settings_file = 'data/settings.json'
class_names, train_indices, test_indices, test_sequence, experiment_id, scale = ut.load_settings(settings_file)

# load strategy data i.e. image sequences
strat_files = glob.glob('data/*.strat')
strats = ut.load_strats(strat_files, test_sequence)

# tutorial images
tutorial_images = ['tutorial_0.jpg', 'tutorial_1.jpg', 'tutorial_2.jpg', 'tutorial_3.jpg']

def initalize_session():
    # create new user session when user visits home page
    session.clear()
    session['name'] = str(uuid.uuid4())
    session['response'] = []
    session['time'] = []
    session['gt_label'] = []
    session['strategy'] = random.choice(strats.keys())
    # different users will see options in different order - order will be consistent for the length of session
    session['button_order'] = random.sample(range(len(class_names)), len(class_names))
    session['current_id'] = 0
    session['experiment_id'] = experiment_id

    num_ims = strats[session['strategy']]['num_train'] + strats[session['strategy']]['num_test']
    session['num_ims'] = num_ims
    session['num_train'] = strats[session['strategy']]['num_train']
    session['num_test'] = strats[session['strategy']]['num_test']

    if 'random' in session['strategy']:
        # random strategies
        session['image_id'] = random.sample(train_indices, session['num_train']) + list(strats[session['strategy']]['test_sequence'])
        session['is_train'] = [0]*num_ims
        session['display_explain_image'] = [0]*num_ims

        for ii in range(session['num_train']):
            session['is_train'][ii] = 1

            if strats[session['strategy']]['display_explain_image']:
                session['display_explain_image'][ii] = 1

    else:
        session['is_train'] = list(strats[session['strategy']]['is_train'])
        session['image_id'] = list(strats[session['strategy']]['image_id'])
        session['display_explain_image'] = list(strats[session['strategy']]['display_explain_image'])


    # can have random images in the test set by specifying the index as -1
    valid_remain_test = list(set(test_indices) - set(session['image_id']))
    valid_remain_test = random.sample(valid_remain_test, len(valid_remain_test))
    vv = 0
    for ii in range(num_ims):
        if session['image_id'][ii] == -1 and session['is_train'][ii] == 0:
            session['image_id'][ii] = valid_remain_test[vv]
            vv += 1

    # add labels to session
    for ii in range(num_ims):
        session['gt_label'].append(images[session['image_id'][ii]]['class_label'])

    session.modified = True


@application.route('/')
def index():
    # Create new session when users visits the homepage
    initalize_session()
    print(session['strategy'])

    params = {}
    params['num_ims'] = session['num_ims']
    params['class_names'] = class_names

    return render_template('index.html', params=params)


@application.route('/tutorial/<im_id>')
def tutorial(im_id):
    params = {}
    params['im_id'] = int(im_id)
    params['tutorial_images'] = tutorial_images

    return render_template('tutorial.html', params=params)


@application.route('/debug')
def debug():
    params = {}
    params['images'] = images
    params['class_names'] = class_names
    params['strats'] = strats

    return render_template('debug.html', params=params)

@application.route('/dashboard')
def dashboard():
    # Display results per strategy - for this experiment
    user_data = list(db.user_results.find({'experiment_id':experiment_id}))

    if len(user_data) == 0:
        return 'No results file exists yet.'
    else:
        strat_names = [uu['strategy'] for uu in user_data]
        test_scores = [uu['score'] for uu in user_data]

        params = {}
        params['num_turkers'] = len(user_data)
        params['strat_names'] = strats.keys()
        params['test_scores'] = [0]*len(strats.keys())
        params['users_per_strat'] = [0]*len(strats.keys())
        params['experiment_id'] = experiment_id

        # compute the per strategy average results
        for jj, ss in enumerate(strats.keys()):
            for ii, rr in enumerate(strat_names):
                if ss == rr:
                    params['test_scores'][jj] += test_scores[ii]
                    params['users_per_strat'][jj] += 1

            if params['users_per_strat'][jj] > 0:
                params['test_scores'][jj] /= params['users_per_strat'][jj]

    return render_template('dashboard.html', params=params)


@application.route('/user_data')
def user_data():
    # Display user results - for this experiment
    user_data = list(db.user_results.find({'experiment_id':experiment_id}))

    if len(user_data) == 0:
        return 'No results file exists yet.'
    else:
        return dumps(user_data)


def save_results(session):
    # Write data for current session to database
    result = {}
    result['completion_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result['strategy'] = session['strategy']
    result['image_id'] = session['image_id']
    result['response'] = session['response']
    result['gt_label'] = session['gt_label']
    result['is_train'] = session['is_train']
    result['time'] = session['time']
    result['mturk_code'] = session['name']
    result['score'] = session['score']
    result['experiment_id'] = session['experiment_id']

    # write to DB
    db.user_results.insert_one(result)


@application.route('/teaching', methods=['GET','POST'])
def teaching():
    # Main function that handles the teaching logic

    # session not initialized - so do it
    if 'response' not in session.keys():
        initalize_session()

    # capture user button presses
    if request.method == 'POST':
        session['response'].append(int(request.form['action']))
        session['time'].append(datetime.datetime.now().strftime("%H:%M:%S"))
        session['current_id'] += 1
        session.modified = True

    # if they are finished, send users to results screen and save results to db
    if len(session['response']) == session['num_ims']:

        # compute score on test set
        num_corr = 0.0
        for rr in range(session['num_train'], session['num_ims']):
            if session['response'][rr] == session['gt_label'][rr]:
                num_corr += 1.0

        params = {}
        params['mturk_code'] = session['name']
        params['score'] = round(100*num_corr / float(session['num_test']),2)
        session['score'] = params['score']
        session.modified = True

        save_results(session)

        return render_template('results.html', params=params)

    # select next image to show
    current_id = session['current_id']
    image_id = session['image_id'][current_id]

    params = {}
    params['image'] = images[image_id]['image_url']
    params['explain_image'] = images[image_id]['explain_url']
    params['label'] = images[image_id]['class_label']
    params['display_explain_image'] = session['display_explain_image'][current_id] and session['is_train'][current_id]
    params['len_resp'] = len(session['response'])
    params['strategy'] = session['strategy']
    params['is_train'] = session['is_train'][current_id]
    params['total_num_ims'] = session['num_ims']
    params['num_train'] = session['num_train']
    params['class_names'] = class_names
    params['training_finished'] = session['num_train'] == session['current_id']
    params['button_order'] = session['button_order']
    params['scale'] = scale

    params['train_feedback'] = False
    if len(session['response']) == (session['num_train']//2):
        params['train_feedback'] = True

    return render_template('teaching.html', params=params)


if __name__ == '__main__':
    #application.run(debug=True, use_reloader=True)
    application.run()
