# Deep learning lab course final project.
# Kaggle whale classification.

# Helper functions for the main keras model.

import datetime
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import keras.utils
import utilities as ut


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
    ut.save_plot(x, ys=losses, xlabel="epoch", ylabel="loss",
                 title=run_name, path=fn_losses)
    ut.save_plot(x, ys=accuracies, xlabel="epoch", ylabel="accuracy",
                 title=run_name, path=fn_accuracies)


def save_learning_curves_2(history, cnn_after, run_name, base_path="plots/"):
    """Saves the data from keras history dict in loss and accuracy graphs to folder
    specified by base_path and run_name."""
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    # losses = {k: history[k] for k in ['loss', 'val_loss']}
    accuracies = {k: history[k] for k in ['val_acc','acc']}
    accuracies = {k: history[k] for k in ['val_acc']}
    
    x = range(len(accuracies['val_acc']))
    # fn_losses = os.path.join(path, "loss.png")
    fn_accuracies = os.path.join(path, "accuracy.png")
    ut.save_plot_2(cnn_after, x, ys=accuracies, xlabel="epoch", ylabel="accuracy",
                 title=run_name, path=fn_accuracies)    
    

def draw_num_classes_graphs():
    print("Will likely not work because")
    print("keras_tools.draw_num_classes_graphs() was not yet adapted")
    print("to the usage of config_dict in keras_model.py")
    """Train network and save learning curves for different values for num_classes."""
    values = [10, 50, 100, 250, 1000, 4000]
    for num_classes in values:
        print("Training model on {} most common classes.".format(num_classes))
        model = create_pretrained_model(num_classes=num_classes)
        histories = train(model, num_classes, epochs=50)
        run_name = get_run_name("{}classes".format(num_classes))
        save_learning_curves(histories, run_name)
        csv_path = os.path.join("plots/", run_name, "data.csv")
        ut.write_csv_dict(histories,
                       keys=['loss', 'acc', 'val_loss', 'val_acc'],
                       filename=csv_path)


def visualize_model(model=None, 
                    filename="InceptionV3_visualization.png",
                    show_shapes=False):
    """
    Write graph visualization of Keras Model to file.
    Default model is InceptionV3
    """
    if model is None:
        model = InceptionV3(weights='imagenet', include_top=False)
    else:
        model = model
    keras.utils.print_summary(model)
    print("---")
    print("len(model.layers)", len(model.layers))
    print("saving graph visualization to file")
    keras.utils.plot_model(model, show_shapes=show_shapes, to_file=filename)
    print("saved graph visualization to file")
    
    for i, layer in enumerate(model.layers):
       print(i, layer.name)    

# get array of model predictions [samples,class-probabilites]    
def get_flow_from_dir(test_dir, target_size=(224, 224)):
    
    batch_size = 16     # used for training as well as validation
    
    test_gen = image.ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    test_flow = test_gen.flow_from_directory(
        test_dir,
        shuffle=False,          
        batch_size = batch_size,     
        target_size = target_size,
        class_mode = None)    # use "categorical" ??
    
    return test_flow 
    
    
# get array of model predictions [samples,class-probabilites]    
def predict_probs(model, test_dir):
    
    batch_size = 16     # used for training as well as validation
    
    if model.name == 'InceptionV3' or model.name == 'Xception' or model.name == 'InceptionResNetV2':
        target_size = (299, 299)
    elif model.name == 'ResNet50' or model.name == 'MobileNet':
        target_size = (224, 224)
    else:
        print("invalid model: ", model.name)
    print("predicting model", model.name) 
    
    test_gen = image.ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    test_flow = test_gen.flow_from_directory(
        test_dir,
        shuffle=False,          
        batch_size = batch_size,     
        target_size = target_size,
        class_mode = None)    # use "categorical" ??
    
    probs = model.predict_generator(test_flow, verbose = 1)
    
    return test_flow, probs

# from probability array get list of model predictions: ordered list of 5 whalenames per image
def get_model_preds(train_dir, probs, threshold = 0):

    max_preds = 5        # number of ranked predictions (default 5)    
    # whale_class_map = (test_flow.class_indices)           # get dict mapping whalenames --> class_no
    class_whale_map = ut.make_label_dict(directory=train_dir)    
    # --> alphanumeric order, see Keras Docu
    
    # probs is array [samples, class_probs] where each col represents on class (col_no = class idx)
    # top_k is array of top 5 class_indicees
    # 
    top_k = probs.argsort()[:, -max_preds:][:, ::-1]
    if threshold == 0:
        model_preds = [([class_whale_map[i] for i in line]) for line in top_k]
    else:
        model_preds = []
        for j, line in enumerate(top_k):
            model_pred = []
            if probs[j][line[0]] < threshold:   # ['new_whale','whale_1','whale_2','whale_3','whale_4']
                model_pred.append = 'new_whale'            
                for i in range(len(line)-1):
                    model_pred.append(class_whale_map[line[i]])
            else: # type ['whale_1','new_whale','whale_2','whale_3','whale_4']
                model_pred.append(class_whale_map[line[0]])   
                model_pred.append('new_whale')
                for i in range(len(line)-2):
                    model_pred.append(class_whale_map[line[i+1]])

            model_preds.append(model_pred)
            
        return(model_preds)
       
    test_preds = [([class_whale_map[i] for i in line]) for line in top_k]
    if test_pres != model_preds:
        print("Error!!!")
    
# return list of true labels of 
def get_true_labels(test_flow, test_csv):
    
    # get list of true labels from csv file: one whalename per image
    true_labels = []
    file_names = []
    if test_csv != '':
        test_list = ut.read_csv(file_name = test_csv)    # list with (filename, whalename)
 
    i = 0   
    for fn in test_flow.filenames:
        if i<3:
            print("fn",fn)
        i=i+1
        offset, directory, filename = fn.split('/')
        file_names.append(filename)
        if test_csv != '':
            whale = [line[1] for line in test_list if line[0]==filename][0]
            true_labels.append(whale) 
            
    return file_names, true_labels

# make prediction of top five candidates, return list of all filenames and 
# corresponding list five top predicted whalenames as well as all probs. 
# true_lables != [] only if test_csv!=''  (validation data)
def compute_preds(model, num_classes, train_dir = "data/model_train", 
                  test_dir = "data/model_valid", test_csv = "data/model_valid.csv"):
    
    test_flow, probs = predict_probs(model, test_dir)   # get array [samples, all classes_probs]

    model_preds = get_model_preds(train_dir, probs) # array [samples, 5 top whale_names]
    
    file_names, true_labels = get_true_labels(test_flow, test_csv)
    
    return file_names, model_preds, true_labels, probs


def compute_bagged_preds(models, num_classes, train_dir = "data/model_train", 
                         test_dir = "data/model_valid", test_csv = "data/model_valid.csv"):
    
    probs_list = []
    for model in models:
        test_flow, probs = predict_probs(model, test_dir)
        probs_list.append(probs)
    mean_probs = np.mean(probs_list, axis=0)

    model_preds = get_model_preds(train_dir, mean_probs)
    
    file_names, true_labels = get_true_labels(test_flow, test_csv)
    
    return file_names, model_preds, true_labels, probs_list
    
    
def write_pred_to_csv(file_names, model_preds, path = "data/submission.csv"):
    csv_list = []
    for i in range(len(model_preds)):
        csv_row = ['','']
        csv_row[0] = file_names[i]
        s = 'new_whale'    # string containing the five whale names separated by blanks
        for j in range(len(model_preds[i])-1):   # run over 5 ordered predictions
            # if j>0:
            s = s + ' '
            s = s + model_preds[i][j]
            # print("next_s", s)
        csv_row[1] = s
        csv_list.append(csv_row)
    # print("csv_list", csv_list)
    print("write csv file")
    ut.write_csv(csv_list, path)
    print("done writing csv file")      

def write_pred_to_csv_new(file_names, model_preds, path = "data/submission.csv"):
    csv_list = []
    for i in range(len(model_preds)):
        csv_row = ['','']
        csv_row[0] = file_names[i]
        # s = 'new_whale'    # string containing the five whale names separated by blanks
        for j in range(len(model_preds[i])):   # run over 5 ordered predictions
            if j>0:
            s = s + ' '
            s = s + model_preds[i][j]
            # print("next_s", s)
        csv_row[1] = s
        csv_list.append(csv_row)
    # print("csv_list", csv_list)
    print("write csv file")
    ut.write_csv(csv_list, path)
    print("done writing csv file") 
    
    
# perform prediction on validation data, compare with true labels and compute acc and MAP    
def compute_map(model_preds, true_labels):
    max_preds = len(model_preds[0])
    print("max_preds", max_preds)
    # print("model predictions: \n", np.array(model_preds)[0:10])
    # print("true labels \n", np.array(true_labels)[0:10])
    
    # compute accuracy by hand
    TP_List = [(1 if model_preds[i][0]==true_labels[i] else 0) for i in range(len(true_labels))]
    acc = np.sum(TP_List) / len(true_labels)
    print("{} true predictions out of {}: accurracy: {} ".format(np.sum(TP_List),len(true_labels),acc))

    MAP = ut.mean_average_precision(model_preds, true_labels, max_preds)
    print("MAP", MAP)
    
    return MAP



if __name__ == "__main__":
    import sys
    if "--visualize_inceptionV3" in sys.argv:
        visualize_model()
