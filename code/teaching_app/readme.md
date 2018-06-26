# Overview
This is a simple Flask based web app that will display either random images to user or images in a sequence specified in the configuration files. When a user visits the site, they will be assigned randomly to one of several possible specified teaching strategies and will be shown images and potentially visual exanations as feedback.  


### Data Format
There are three types of files that are used to configure the images that are shown to the users. These files should be placed in `data/`.

1) List of images - teaching_images.json  
This is a list of dictionaries where each entry contains the image url, the url of an explanation image (can be blank), and a class label from 0 to the number of classes (i.e. this can be multi-class).  
Images can be hosted on your own website or some external service such as Amazon S3.
Here is an example entry for one image:    
```
{
  'class_label': 0,
  'image_url': 'https://blaablaablaa.com/image.jpg',
  'explain_url': 'https://blaablaablaa.com/explain_image.jpg'
}
```

2) Settings - settings.json  
This contains a list of class names, the indices of the train (train_indices) and testing images (test_indices), and the indices of the sequence of test images that will be shown (test_sequence) - these indices reference the id in the image list in teaching_images.json. If there is a `-1` in test_sequence a random image from the unseen test set will be chosen for that location. There is also a `scale` value which allows images to be displayed bigger in the user interface. The number of class names should be the same as the number of unique classes in teaching_images.json. You can also specify an experiment id number (experiment_id) to keep track of results.    

3) Teaching sequence files e.g. strategy_0.strat, strategy_1.strat, ...  
These files can have any name as long as they end with .strat.    
im_ids is the list of images that will be displayed during training, selected from teaching_images.json - test images are specified in settings.json to ensure that all strategies use the same test images.
For example, the following strategy file would display three images (2, 1, and 0), with an explanation image after each.
```
{
    "num_train": 3,
    "display_explain_image": [1, 1, 1],  
    "im_ids": [2, 1, 0]
}
```

You can also specify random strategy files e.g. random_image.strat and random_image_with_explain.strat. These contains two entries - the number of training images, and whether to display the explanation image:  
```
{
    "num_train": 3,
    "display_explain_image": 1
}
```

### General Notes
Will only save user data if they complete the task.  
Good idea to keep all images the same size where possible.  
Change `config.py` to point to database and set the secret key.  


## Deployment
For testing the app can be deployed locally, but to run experiments on the web it must be accessible online.


### Set up Database on mlab  
We need to store our results in a database.  
Create account at `https://mlab.com`  
When logged in go to MongoDB Deployments, click Create new, choose Sandbox (free), click Continue (bottom right), select US region, give it a name.  
Once created you need to create a new users account for the database. To do this click on database name, Users, and add database user.  
Finally set the MonogoDB URI in `config.py`.


### Locally
Assuming you have Flask installed on your machine (it comes bundled in the Anaconda installation), just download the repository and run `python application.py`.  
As configured, there must be a database installed and a secret key set in `config.py` or the app will throw an error.  


### Heroku
Alternatively, you can deploy the app on the web so it's accessible to others using a service such as Heroku.  
First, register for a free heroku account at `https://www.heroku.com/`.   
Download Heroku CLI for your machine - `https://devcenter.heroku.com/articles/heroku-cli`.   
Download the code from this repository and save it on your computer.  

Create new a Heroku webapp - this will be the website that hosts our application  
```
heroku login
heroku create <app-name> e.g. heroku create machine-teaching-demo (you won't be able to use this app name as I already have it)
```

Create database - we use the database we created on mlab  
`heroku addons:create mongolab:sandbox --app <app-name>`

Create new git repo - Heroku uses git for deploying app   
```
cd my-project/
git init
heroku git:remote -a <app-name> e.g. heroku git:remote -a machine-teaching-demo
```

Deploy - add the code to the repository and psuh to Heroku   
```
git add .
git commit -am "first commit"
git push heroku master
```

For existing repositories, simply add the Heroku remote  
```
heroku git:remote -a machine-teaching-demo
```

View app at  
```
https://<app-name>.herokuapp.com/ e.g. https://your-machine-teaching-demo.herokuapp.com/
```

### AWS Elastic Beanstalk  
Create Access Key  
AWS Console -> My Security Credentials  

Configure CLI  
http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html  

Flask details  
http://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html  

```
eb init -p python2.7 image-quiz  # create app  
eb create image-quiz-production  # create environment  
eb deploy  # update app every time you make changes  
eb terminate image-quiz-production  # deletes the app
```

### View Results
Go to `https://<app-name>/debug` to see if data loaded correctly.    

Go to `https://<app-name>/user_data`, copy the text and save it in `results.txt`.  
```
import json
results_file = 'results.txt'
with open(results_file) as f:   
    user_data = json.load(f)

print len(user_data), 'users completed the task'
print user_data[0]['mturk_code']
print user_data[0]['strategy']
print user_data[0]['response']
print user_data[0]['gt_label']
```

Can also go to `https://<app-name>/dashboard` to view the test set summary live.  

Alternatively, you can load the results directly from the database:  
```
from pymongo import MongoClient  
import config  

client = MongoClient(config.MONGO_DB_STR)  
db = client.get_default_database()  
experiment_of_interest = 1

user_data = list(db.user_results.find())

# This will delete all the data in the DB - warning only do this if everything is backed up  
# db.user_results.drop()

# This will delete all the entries for a specific experiment.  
# db.user_results.delete_many({'experiment_id':experiment_of_interest})  
```