# Convolutional Neural Networks on the FashionMNIST dataset using Tensorflow

This tutorial goes through the creation of Convolutional Neural Networks(CNNs) using tensorflow and shows how we can package that and train on the cloud. To demonstrate this we will use the google cloud tensorflow templates that makes use of the fashionMNIST dataset. This tutorial is a follow-up of the first tutorial that used Deep Neural Networks and it is mostly for emphasis. 

The fashionMNIST dataset has exactly the same structure as the MNIST dataset with 60 000 training, 10 000 testing examples, 10 classes and 28×28 grayscale/single channel images. The fashion MNIST dataset will be used to classify between different clothing items. Before beginning, we will give a brief introduction of CNNs, best use cases then we will get our hands dirty with some code.

Basics and Types of Layers:
Convolutional Layers
Each layer of the network consists of filter maps. Each filter detects certain features in the image. Each filter slides over the image and uncovers different features. 

Pooling Layers
This layer decreases the dimensionality of the image inorder to increase the performance of the network and also reduce over-fitting. The pooling layer always comes after the convolutional layers.

Fully-connected layer
This layer has all nodes connected to the nodes in the previous layer. This is a classification layer which receives the features extracted by the convolutional and the pooling layers.

The first thing we need to do is to get our data into a Google cloud storage bucket. For this, we need to have an active project on Google cloud. 

Tensorflow provides some great functions to perform all necessary operations so you only have to worry about the performance of the model. If not, it's insanely easy to create one. Open the Google cloud console and press 'select a project' on the tool bar. It will allow you to either choose an existing project or to create a new project. Once we have our project selected or created, next thing is to create a bucket. On the left hand corner of the Google cloud console, there are 3 horizontal lines. Press the horizontal lines and scroll down until storage. This will then allow you to create a bucket. Once we have our bucket, we press on add files and load our data. For this tutorial I downloaded the fashionMNIST to my local machine then uploaded it to the Google cloud storage. 

Trainer Package:
In order to train on Google cloud, there is a convention. We have to create a trainer package. Since we are using google cloud templates, everything is already done for us. Our trainer package in this case consists of a model.py, task.py and util.py. 

Model.py consists of the model, in this case it is the CNN. In the first tutorial our trainer package had more files. Some of the files that were there were the metadata and the feature engineering file. The reason we don't have them is because CNNs extract features from data while training, so there is no need for manual extraction of features. This is the reason why CNNs work so great for image and sound data.

Our model.py consists of our model. So our CNN is built in our model.py. The utils.py consists of our filenames, and how to unzip them inorder to get them to a usable state. Lastly, the task.py as per the norm is the file you that we call when we want to run the code. This is the file that calls all the relevant files. This file parses parameters like the location of data and hidden layers. It also consists of all the parameters of the model, like the learning rate, dropout rates, number of buckets etc. .  It also calls the training logic in model.py.

In this case, I did put location of data in task.py as default. This way we do not have to input location when we train. You can however set data location as required then input it in terminal when we train. 

Training Locally:
It is always advisable to train locally before training on AI-platform. If the code does run, then we know that is likely to also run on the AI-platform. To train our code locally we run the following code block. 
 gcloud ai-platform local train     
 --module-name trainer.task     
 --package-path trainer/     
 --job-dir $MODEL_DIR 
 
This is very minimal, with trainer.task referring to our task.py, trainer as the trainer package and the job directory is where the output should be stored after training. Do note that there are many other arguments that can be included.

Training on AI-Platform
Now that our package has trained locally with no errors, we  then proceed to train on the cloud. The following command is the command we use.

gcloud ai-platform jobs submit training $JOB_NAME \
   --job-dir $OUTPUT_PATH \
   --runtime-version 1.10 \
   --module-name trainer.task \
   --package-path trainer/ \
   --region $REGION \
   -- \
   --train-files $TRAIN_DATA \
   --eval-files $EVAL_DATA \
   --train-steps 1000 \
   --eval-steps 100 \
   --verbosity DEBUG
   
   
with job name being the name of the job. The job name has to be unique so if you rerun your job you always have to change the name. I usually work around this by placing a number at the end and simply increment the number for lack of creativity. Output path is the location to store 

Once this is run, it gives you an opportunity to track your logs in terminal or on Google cloud. This allows you to monitor your logs, see when there are any warnings to be weary about and also see if your job fails and the reason why. 
