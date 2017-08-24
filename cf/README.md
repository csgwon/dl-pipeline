# DL Pipeline using CloudFormation

### Introduction
A CloudFormation script is provided, which will set up the AWS environment.  It is a strawman architecture which will allow for a proof-of-concept deployed system, setting up your VPC, a single subnet, the web server, an S3 bucket with a Lambda trigger.  The remaining steps required are detailed below.


### Setting up EC2

Log into your EC2 instance (username: ```ubuntu```).  When launching the CloudFormation stack, it should have also asked you for the SSH keys that you will be using, which you can pass in with the ```-i``` option.

After logging into your instance, clone the repository and train the models (~15 min per model).  Both need to be run for the server to operate

```sh
$ git clone https://github.com/csgwon/dl-pipeline.git
$ cd dl-pipeline/train
$ python name_test.py
$ python tf_name_test.py
```

As root, add the following lines to ```/etc/apache2/sites-enabled/000-default.conf``` after ```DocumentRoot /var/www/html```:
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

Also as root, copy over the ```flaskapp``` directory and the trained models into the ```/var/www/html``` area, and then restart apache:
```sh
$ cd /var/www/html
$ cp -rp ~ubuntu/dl-pipeline/flaskapp .
$ cd flaskapp
$ cp ~ubuntu/dl-pipeline/train/*charcnn.* .
$ apachectl restart
```

At this point, your web server will be ready to go.  Add in objects into the S3 bucket (you should have specified a unique bucket name when launching the CloudFormation stack) using AWS CLI or the console.  Objects need to be prefixed with ```name_``` and suffixed with ```.txt```.  After a few seconds, a new file called ```out_name_X.txt``` should be created, where ```X``` is whatever you've chosen to call the file.  This output file will have the predicted nationality of name inside your ```name_X.txt``` file.

This can also be extended for other capabilities as well.  I have tested this with [SSD in Tensorflow](https://github.com/balancap/SSD-Tensorflow.git) and successfully returned annotated images in the same manner.
