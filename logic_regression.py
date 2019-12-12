import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.misc
from PIL import Image
from scipy import ndimage
from Ir_utils import load_dataset
import skimage
#loading the data
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()
print(train_set_x_orig.shape)
print(train_set_y.shape)
print(test_set_x_orig.shape)
print(train_set_y.shape)
print(classes)

#Example of picture
index =106
plt.imshow(train_set_x_orig[index])
print("y="+str(train_set_y[:,index])+",it is a"+classes[np.squeeze(train_set_y[:,index])].decode("utf-8"))

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

def sigmod(z):

    s = 1 / (1 + np.exp(-z))


    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0

    return w,b

def propagate(w,b,X,Y):
     m = X.shape[1]

     A = sigmod(np.dot(w.T,X) + b)
     cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

     dw = 1 / m * np.dot(X,(A-Y).T)
     db = 1 / m * np.sum(A - Y)

     cost = np.squeeze(cost)
     grads = {"dw":dw,
              "db":db}
     return grads,cost
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):

    costs = []

    for i in range(num_iterations):

        grads,cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        #print the cost every 100 training examples

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i:%f"%(i,cost))

    params = {"w":w,
              "b":b}

    grads = {
        "dw":dw,
        "db":db}

    return params,grads,costs

def predict(w,b,X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmod(np.dot(w.T,X) + b)

    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost =False):

    w,b = initialize_with_zeros(X_train.shape[0])

    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)


    print("train accuracy {}%".format(100-np.mean(np.abs(Y_prediction_train - Y_train)) * 100) )
    print("test accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations,
    }
    return d


d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.005,print_cost=True)

my_image = "cat_in_iran.jpg"

fname = "images/"+my_image
image = np.array(plt.imread(fname))
my_image = scipy.misc.imresize(image,size = (num_px,num_px)).reshape((1,num_px*num_px*3)).T
#my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"],d["b"],my_image)

print(plt.imshow(image))
print("y ="+str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a " + classes[int(np.squeeze(my_predicted_image)),].decode('utf-8'))

my_image1 = "my_image.jpg"

fname = "images/"+my_image1
image = np.array(plt.imread(fname))
my_image1 = scipy.misc.imresize(image,size = (num_px,num_px)).reshape((1,num_px*num_px*3)).T
#my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image1 = predict(d["w"],d["b"],my_image1)

print(plt.imshow(image))
print("y ="+str(np.squeeze(my_predicted_image1)) + ", your algorithm predicts a " + classes[int(np.squeeze(my_predicted_image1)),].decode('utf-8'))

my_image1 = "my_image2.jpg"

fname = "images/"+my_image1
image = np.array(plt.imread(fname))
my_image1 = scipy.misc.imresize(image,size = (num_px,num_px)).reshape((1,num_px*num_px*3)).T
#my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image1 = predict(d["w"],d["b"],my_image1)

print(plt.imshow(image))
print("y ="+str(np.squeeze(my_predicted_image1)) + ", your algorithm predicts a " + classes[int(np.squeeze(my_predicted_image1)),].decode('utf-8'))

my_image1 = "cat.jpg"

fname = "images/"+my_image1
image = np.array(plt.imread(fname))
my_image1 = scipy.misc.imresize(image,size = (num_px,num_px)).reshape((1,num_px*num_px*3)).T
#my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image1 = predict(d["w"],d["b"],my_image1)

print(plt.imshow(image))
print("y ="+str(np.squeeze(my_predicted_image1)) + ", your algorithm predicts a " + classes[int(np.squeeze(my_predicted_image1)),].decode('utf-8'))

