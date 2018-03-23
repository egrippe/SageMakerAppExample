import gzip
import io
import json
import boto3
import pickle
import numpy
import matplotlib.pyplot as plt


# load the mnist data from file
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


# Simple function to create a csv from our numpy array
def np2csv(arr):
    csv = io.BytesIO()
    numpy.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


# create a sagemaker client
client = boto3.client('sagemaker-runtime')

# specify an example to classify
example = 30

# convert the image vector to csv
payload = np2csv(train_set[0][example:example+1])

# classify image using sagemaker endpoint
response = client.invoke_endpoint(EndpointName='kmeans-2018-03-23-08-22-27-189',
                                  ContentType='text/csv',
                                  Body=payload)

# print result of prediction
result = json.loads(response['Body'].read().decode())
print(result)


def show_digit(img, caption=''):
    imgr = img.reshape((28, 28))
    plt.imshow(imgr, cmap='gray')
    plt.title(caption)
    plt.show()


# show the example we just classified
show_digit(train_set[0][example], 'This is a {}'.format(train_set[1][example]))
