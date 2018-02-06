# Deep learning lab course final project.
# Kaggle whale classification.

# Helper functions for the main keras model.

import datetime
import os
from keras.applications.inception_v3 import InceptionV3
import keras.utils
import utilities as ut


def get_run_name(additional=""):
    if additional != "":
        additional = "_" + additional
    return "run-{}{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
        additional)


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


def draw_num_classes_graphs():
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



def visualize_model(model=None, filename="InceptionV3_visualization.png"):
    """
    Write graph visualization of Keras Model to file.
    Default model is InceptionV3
    """
    if model is None:
        p_model = InceptionV3(weights='imagenet', include_top=False)
    else:
        p_model = model
    keras.utils.plot_model(p_model, to_file=filename)

    
def print_model_test_info(model, num_classes):
    # try to verify on test data --> no success so far
    
    # use all training data of the first num_classes whales as test data.
    # no good practice, but all training data have been augmented, so at least some indication
    # about predictive power of model
    
    all_train_dir = "data/train"     # directory with original kaggle training data
    all_train_csv = "data/train.csv" # original kaggle train.csv file    
    test_dir = "data/model_test"
    test_csv = "data/model_test.csv"
    batch_size = 16     # used for training as well as validation
    
    num_train_imgs, num_valid_imgs = ut.create_small_case(
        sel_whales = np.arange(1,num_classes+1),  # whales to be considered
        all_train_dir = all_train_dir,
        all_train_csv = all_train_csv,
        train_dir = test_dir,
        train_csv = test_csv,
        valid_dir = None,     # no validation, copy all data into test_dir "data/model_test"
        valid_csv = None,
        train_valid = 1.,
        sub_dirs = True) 

    test_gen = image.ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    test_flow = test_gen.flow_from_directory(
        test_dir,
        # color_mode = "grayscale",
        batch_size = batch_size,     
        target_size = (299,299),
        class_mode = None)    # use "categorical" ??
    
    preds = model.predict_generator(test_flow, verbose = 1)

    whale_class_map = (test_flow.class_indices)           # get dict mapping whalenames --> class_no
    class_whale_map = ut.make_label_dict(directory=test_dir) # get dict mapping class_no --> whalenames
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
    test_list = ut.read_csv(file_name = test_csv)    # list with (filename, whalename)
    true_labels = []
    for fn in test_flow.filenames:
        offset, filename = fn.split('/')
        whale = [line[1] for line in test_list if line[0]==filename][0]
        true_labels.append(whale)

    print("model predictions: \n", np.array(model_preds)[0:10])
    print("true labels \n", np.array(true_labels)[0:10])
    
    # compute accuracy by hand
    TP_List = [(1 if model_preds[i][0]==true_labels[i] else 0) for i in range(len(true_labels))]
    acc = np.sum(TP_List) / len(true_labels)
    print("{} true predictions out of {}: accurracy: {} ".format(np.sum(TP_List),len(true_labels),acc))

    MAP = ut.mean_average_precision(model_preds, true_labels, max_preds)
    print("MAP", MAP)

    # for comparison compute MAP generated by Dummy model
    Dummy_map = np.mean([ut.Dummy_MAP(probs = 'weighted', distributed_as = train_csv, 
                                      image_no = len(test_list)) for i in range(5)])
    print("MAP of Dummy Model averaged over 5 runs: ", Dummy_map)
    return MAP, Dummy_map