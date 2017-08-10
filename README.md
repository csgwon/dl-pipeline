# DL Pipeline

### Introduction
Brief tutorial on developing a pipeline for deploying deep learning capabilities onto AWS.  Suggestions are welcome - this is a work in progress.  The steps which I'll go through are as follows:
- Deploying deep learning algorithm on EC2
- Setting up AWS lambda to read from and write to S3

It's assumed that you have set up your VPCs, security groups, IAM roles, and environment in general.


### Setting up EC2

I started with the Ubuntu 16.04 LTS (HVM) AMI running on a t2.xlarge (probably overkill for my example), and the size will depend on the demands of your model.  For exposing the web service, I used Apache Web Server with mod_wsgi and [Flask](http://flask.pocoo.org/) - thanks to [Data Science Bytes](http://www.datasciencebytes.com/bytes/2015/02/24/running-a-flask-app-on-aws-ec2/) for their tutorial on this.  Although you can follow their instructions, I include the steps that I followed here to get things installed for completeness:
```sh
$ sudo apt-get -y update
$ sudo apt-get -y install apache2 libapache2-mod-wsgi
$ sudo apt-get -y instasll python python-pip
$ sudo pip install flask
```
To allow for dynamic content onto the page, add the following lines to ```/etc/apache2/sites-enabled/000-default.conf``` after ```DocumentRoot /var/www/html```:
```xml
        WSGIDaemonProcess flaskapp threads=1
        WSGIScriptAlias / /var/www/html/flaskapp/flaskapp.wsgi

        <Directory flaskapp>
            WSGIProcessGroup flaskapp
            WSGIApplicationGroup %{GLOBAL}
            Order deny,allow
            Allow from all
        </Directory>
```
```flaskapp``` can be replaced with whatever you'd like to call your application.  In the ```/var/www/html``` area, create the ```flaskapp``` directory, which is where your python code will eventually go.  You'll need to add the ```flaskapp.wsgi``` file containing the following:
```python
import sys
sys.path.insert(0,'/var/www/html/flaskapp')
from flaskapp import app as application
```
### Train your favorite Deep Learning algorithm
Begin with your favorite deep learning algorithm.  For starters, I borrowed one from a [JoostWare] (https://github.com/joosthub/pytorch-nlp-tutorial).  Whereas I'm a big advocate of PyTorch, it would mysteriously hang on me when deployed with Apache with mod_wsgi (works fine locally with Flask alone).  Because of that, I created a Keras version as well.  I'm including both in case someone could shed insights as to why the PyTorch one hangs.  The versions I'm using are PyTorch 0.2.0.1, Tensorflow 1.2.1, and Keras 2.0.6.

Each model takes about ~15 min to train on a t2.xlarge (~3 min on a p2.xlarge) and achieves ~93-97% accuracy on the test set.  To generate models, go into the ```train/``` directory and run the following for Tensorflow:
```sh
$ python tf_name_test.py
```
which will create ```tf_charcnn.h5``` or for PyTorch:
```sh
$ python name_test.py
```
which will create ```charcnn.pth```.

### Deploying your Deep Learning algorithm onto EC2
Copy all the files from ```flaskapp/``` and the ```tf_charcnn.h5``` model into ```/var/www/html/flaskapp``` (and ```charcnn.pth``` if you're testing out the PyTorch version). ```flaskapp.py``` contains the web service calls, which can be modified as necessary.  In its present state, it contains the following:
```python
from flask import Flask, render_template
import namecnn
import tf_predict
from tools import *

app = Flask(__name__)

@app.route('/name/<input_str>')
def predict(input_str):
    return namecnn.predict(input_str)

@app.route('/name_tf/<input_str>')
def predict_tf(input_str):
    return tf_predict.predict(input_str)

if __name__ == "__main__":
    app.run()
```
Then run the following command to start your web server:
```sh
$ apachectl restart
```
and your web server should be live.  Look at ```/var/log/apache2/error.log``` should something require debugging.  At this point, you should be able to test your code (assuming you can get past the security group) by using your EC2 external DNS hostname (e.g. ```http://ec2-XXX-XXX-XXX-XXX.compute-1.amazonaws.com/name_tf/sanchez```) and substuting "sanchez" for the name of your choice. 

### Setting up Lambda
I'm assuming you are familiar with creating a Lambda function.  For my example, I created one based on the ```s3-get-object-python``` blueprint.  Note, you will also need to create an S3 bucket for this example.  Also if you are working within a VPC, be sure to create a VPC endpoint so that you can communicate with S3 (the reason explained [here](https://aws.amazon.com/blogs/aws/new-vpc-endpoint-for-amazon-s3/)).  I used the "ObjectCreated" trigger, adding a prefix condition of ```name_``` and a suffix of ```.txt``` to control kicking off the Lambda function (particularly since I was adding the results back into the same S3 bucket).  Much of the boiler plate exists in the blueprint, but my final Lambda script looks as follows:

```python
from __future__ import print_function

import json
import urllib
import boto3

print('Loading function')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.unquote_plus(event['Records'][0]['s3']['object']['key'].encode('utf8'))
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        name = response['Body'].read().strip()
        nationality = urllib.urlopen("http://ec2-XXX-XXX-XXX-XXX.compute-1.amazonaws.com/name_tf/"+name).read()
        s3.put_object(Bucket=bucket, Key='out_'+key, Body=nationality)
        return nationality
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
```
When you add a file into your S3 bucket called ```name_X.txt``` with a name in it, it will return a file called ```outname_X.txt``` with a guess to the nationality.
