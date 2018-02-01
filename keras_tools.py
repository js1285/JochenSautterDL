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

