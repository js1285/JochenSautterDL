# Deep learning lab course final project.
# Kaggle whale classification.

import os
import sys
import pickle
import csv
import numpy as np
from scipy.misc import imread
import shutil
import matplotlib.pyplot as plt
import datetime

import h5py
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

from utilities import *


# global variables
all_train_dir = "data/train"     # directory with original kaggle training data
all_train_csv = "data/train.csv" # original kaggle train.csv file
train_dir = "data/model_train"
train_csv = "data/model_train.csv"
valid_dir = "data/model_valid"
valid_csv = "data/model_valid.csv"

num_classes = 7     # number of whales to be considered (in order of occuurence)
max_preds = 5       # number of ranked predictions (default 5)
batch_size = 16     # used for training as well as validation
train_valid = 0.8   # ratio training / validation data

# Use pretrained model as described in https://keras.io/applications/


# create training environment for training data
def prepare_environment(num_classes=num_classes):
    num_train_imgs, num_valid_imgs = create_small_case(
       sel_whales = np.arange(1,num_classes+1),  # whales to be considered
       all_train_dir = all_train_dir,
       all_train_csv = all_train_csv,
       train_dir = train_dir,
       train_csv = train_csv,
       valid_dir = valid_dir,
       valid_csv = valid_csv,
       train_valid = train_valid,
       sub_dirs = True)


def print_data_info():
    # define image generator
    train_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale = 1./255,   # redundant with featurewise_center ? 
        # preprocessing_function=preprocess_input, not used in most examples
        # horizontal_flip = True,    # no, as individual shapes are looked for
        fill_mode = "nearest",
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=30)
    
    # train the model on the new data for a few epochs
    train_flow = train_gen.flow_from_directory(
        train_dir,
        # save_to_dir = "data/model_train/augmented",    
        # color_mode = "grayscale",
        target_size = (299,299),
        batch_size = batch_size, 
        class_mode = "categorical")

    valid_gen = image.ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    valid_flow = valid_gen.flow_from_directory(
        valid_dir,
        target_size = (299,299),
        class_mode = "categorical")

    whale_class_map = (train_flow.class_indices)           # get dict mapping whalenames --> class_no
    class_whale_map = make_label_dict(directory=train_dir) # get dict mapping class_no --> whalenames
    print("whale_class_map:")
    print(whale_class_map)
    print("class_whale_map:")
    print(class_whale_map)
    #print("num_train_imgs:")
    #print(num_train_imgs)
    

def create_pretrained_model(two_layers=True, num_classes=num_classes):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # fully-connected layer on top
    x = Dense(1024, activation='relu')(x)
    if two_layers:
        # new added as https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
    # logistic layer
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    # metrics='accuracy' causes the model to store and report accuracy (train and validate)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def unfreeze_cnn_layers(model):
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    print("unfrozen 2 top CNN layers")
    return model

def train_and_save(model, epochs=20, cnn_epochs=0, cross_validation_iterations=1, 
                   num_classes=num_classes, model_save_fn=None):
    # train in cross validation loop
    histories = []
    for i in range(cross_validation_iterations):
        # create new environment with new random train / valid split
        num_train_imgs, num_valid_imgs = create_small_case(
        sel_whales = np.arange(1,num_classes+1),  # whales to be considered
        all_train_dir = all_train_dir,
        all_train_csv = all_train_csv,
        train_dir = train_dir,
        train_csv = train_csv,
        valid_dir = valid_dir,
        valid_csv = valid_csv,
        train_valid = train_valid,
        sub_dirs = True) 

        # define image generator
        train_gen = image.ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            rescale = 1./255,   # redundant with featurewise_center ? 
            # preprocessing_function=preprocess_input, not used in most examples
            # horizontal_flip = True,    # no, as individual shapes are looked for
            fill_mode = "nearest",
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30)

        train_flow = train_gen.flow_from_directory(
            train_dir,
            # save_to_dir = "data/model_train/augmented",    
            # color_mode = "grayscale",
            target_size = (299,299),
            batch_size = batch_size, 
            class_mode = "categorical")

        valid_gen = image.ImageDataGenerator(
            rescale = 1./255,
            fill_mode = "nearest")

        valid_flow = valid_gen.flow_from_directory(
            valid_dir,
            target_size = (299,299),
            class_mode = "categorical") 

        hist = model.fit_generator(
            train_flow, 
            steps_per_epoch = num_train_imgs//batch_size,
            verbose = 2, 
            validation_data = valid_flow,   # to be used later
            validation_steps = num_valid_imgs//batch_size,
            epochs=epochs)
        
        histories.append(hist.history)

        if cnn_epochs > 0:
            model = unfreeze_cnn_layers(model)
            hist_cnn = model.fit_generator(
                train_flow, 
                steps_per_epoch = num_train_imgs//batch_size,
                verbose = 2, 
                validation_data = valid_flow,   # to be used later
                validation_steps = num_valid_imgs//batch_size,
                epochs=cnn_epochs)
            
        histories.append(hist_cnn.history)  
        # appears as new, seperate run: 
        # TBD: concatenate new values to all lists in two "history" dictionaries
        # or workaround: introduce own global variable for history to be manipulated easily
            

    if model_save_fn !=None:
        model.save(model_save_fn)
    return histories
    

def print_model_test_info(model):
    # try to verify on test data --> no success so far
    
    # use all training data of the first num_classes (7) whales as test data.
    # no good practice, but all training data have been augmented, so at least some indication
    # about predictive power of model
    test_dir = "data/model_test"
    test_csv = "data/model_test.csv"
    num_train_imgs, num_valid_imgs = create_small_case(
        sel_whales = np.arange(1,7+1),  # whales to be considered
        all_train_dir = all_train_dir,
        all_train_csv = all_train_csv,
        train_dir = test_dir,
        train_csv = test_csv,
        valid_dir = None,     # no validation, copy all data into test_dir "data/model_test"
        valid_csv = None,
        train_valid = 1.,
        sub_dirs = True) 
    
    # for test Purposes !!!

    # valid_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale = 1./255,
        # preprocessing_function=preprocess_input,   # model specific function
        fill_mode = "nearest")

    test_flow = test_gen.flow_from_directory(
        test_dir,
        # color_mode = "grayscale",
        batch_size = batch_size,     
        target_size = (299,299),
        class_mode = "categorical")    # use "None" ??
    
    preds = model.predict_generator(test_flow, verbose = 1)

    whale_class_map = (test_flow.class_indices)           # get dict mapping whalenames --> class_no
    class_whale_map = make_label_dict(directory=test_dir) # get dict mapping class_no --> whalenames
    print("whale_class_map:")
    print(whale_class_map)
    print("class_whale_map:")
    print(class_whale_map)
    print("preds.shape:")
    print(preds.shape)
    print("preds[:10]")
    print(preds[:10])
    
    # get list of model predictions: one ordered list of maxpred whalenames per image
    top_k = preds.argsort()[:, -max_preds:][:, ::-1]
    model_preds = [([class_whale_map[i] for i in line]) for line in top_k]  

    # get list of true labels: one whalename per image
    test_list = read_csv(file_name = test_csv)    # list with (filename, whalename)
    true_labels = []
    for fn in test_flow.filenames:
        offset, filename = fn.split('/')
        whale = [line[1] for line in test_list if line[0]==filename][0]
        true_labels.append(whale)

    print("model predictions: \n", np.array(model_preds)[0:10])
    print("true labels \n", np.array(true_labels)[0:10])
    
    # compute accuracy by hand
    TP_List = [(1 if model_preds[i,0]==true_labels[i] else 0) for i in range(len(true_labels))]
    acc = len(true_labels)/np.sum(TP_List)
    print("{} true predictions out of {}: accurracy: {} ".format(np.sum(TP_List),len(true_labels),acc))

    MAP = mean_average_precision(model_preds, true_labels, max_preds)
    print("MAP", MAP)

    for i in range(10):    # run Dummy MAP generator 
        Dummy_map = Dummy_MAP(probs = 'weighted', distributed_as = train_csv, image_no = len(test_list))
        print("Run Nr. {}, Dummy MAP weighted {}".format(i, Dummy_map))

    # MAP only slightly higher than averag dummy MAP


def save_learning_curves(history, run_name, base_path="plots/"):
    """Saves the data from keras history dict in loss and accuracy graphs to folder
    specified by base_path and run_name."""
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    losses = {k: history[k] for k in ['loss', 'val_loss']}
    accuracies = {k: history[k] for k in ['acc', 'val_acc']}
    x = range(len(losses['loss']))
    fn_losses = os.path.join(path, "loss.png")
    fn_accuracies = os.path.join(path, "accuracy.png")
    save_plot(x, ys=losses, xlabel="epoch", ylabel="loss", title=run_name, path=fn_losses)
    save_plot(x, ys=accuracies, xlabel="epoch", ylabel="accuracy", title=run_name, path=fn_accuracies)


def draw_num_classes_graphs():
    """Train network and save learning curves for different values for num_classes."""
    values = [10, 50, 100, 250, 1000]
    for num_classes in values:
        print("Training model on {} most common classes.".format(num_classes))
        model = create_pretrained_model(num_classes=num_classes)
        histories = train_and_save(model, epochs=1, num_classes=num_classes)
        run_name = "run-{}_{}classes".format(num_classes,
                                             datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        save_learning_curves(histories[0], run_name)
        csv_path = os.path.join("plots/", run_name, "data.csv")
        write_csv_dict(histories[0],
                       keys=['loss', 'acc', 'val_loss', 'val_acc'],
                       filename=csv_path)
    


def main():
    print("Run complete script: Printing data info, train, print test results.")
    # prepare_environment()
    print_data_info()
    model = create_pretrained_model()
    histories = train_and_save(model, epochs=2)
    print(histories)
    run_name = "run-{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    save_learning_curves(histories[0], run_name)
    write_csv_dict(histories[0], keys=['loss', 'acc', 'val_loss', 'val_acc'], filename=run_name + '.csv')
    print_model_test_info(model)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
        exit()
    if "--prepare-only" in sys.argv:
        prepare_environment()
        exit()
    if "--class-graph" in sys.argv:
        draw_num_classes_graphs()
        exit()
    print("given command line options unknown.")
    
