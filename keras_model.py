# Deep learning lab course final project.
# Kaggle whale classification.

import os
import sys
import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

import utilities as ut
import keras_tools as tools

# Use pretrained model as described in https://keras.io/applications/

def create_pretrained_model(two_layers=True, num_classes=10, base_mod='InceptionV3'):

    if base_mod == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)  
    elif base_mod == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False)
    elif base_mod == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif base_mod == 'MobileNet':
        base_model = MobileNet(weights='imagenet', input_shape=(224, 224,3), include_top=False)
    elif base_mod == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    else:
        print("invalid model: ", base_mod)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    if two_layers:
        # recommended by
        # https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)

    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)    
    
    # train only the top classifier layers
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to
    # non-trainable)
    # metrics='accuracy' causes the model to store and report accuracy (train
    # and validate)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.name = base_mod    # to identify model in "unfreeze_layers() and "train()" function
    print("successfuly created model: ", model.name)    
    
    return model


def unfreeze_cnn_layers(model):
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 2 layer blocks and unfreeze the rest:
    if model.name == 'InceptionV3':
        cut_off = 249
    elif model.name == 'Xception':
        cut_off = 116
    elif model.name == 'ResNet50':
        cut_off = 152      
    else:
        print("invalid model: ", model.name)

    for layer in model.layers[:cut_off]:
       layer.trainable = False
    for layer in model.layers[cut_off:]:
       layer.trainable = True        
        
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    
    print("\n ****** unfrozen 2 top CNN layers ******")
    return model


def train(model, num_classes, epochs=20, cnn_epochs = 0, save_path=None, batch_size=16,  
          train_dir="data/model_train", valid_dir="data/model_valid", train_valid_split=0.7):

    if model.name == 'InceptionV3' or model.name == 'Xception' or model.name == 'InceptionResNetV2':
        target_size = (299,299)
    elif model.name == 'ResNet50' or model.name == 'MobileNet':
        target_size = (224,224)
    else:
        print("invalid model: ", model.name)
    print("training model", model.name)    
    
    # create new environment with new random train / valid split
    num_train_imgs, num_valid_imgs = ut.create_small_case(
        sel_whales=np.arange(1, num_classes+1),
        all_train_dir = "../data/train",
        all_train_csv = "../data/train.csv",
        train_dir=train_dir,
        valid_dir=valid_dir,
        train_valid=train_valid_split,
        sub_dirs=True)
            
    train_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,   # redundant with featurewise_center ? 
        # preprocessing_function=preprocess_input, not used in most examples
        # horizontal_flip = True,    # no, as individual shapes are looked for
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        # save_to_dir="data/model_train/augmented",    
        # color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size, 
        class_mode="categorical")

    valid_gen = image.ImageDataGenerator(
        rescale=1./255,
        fill_mode="nearest")

    valid_flow = valid_gen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        class_mode="categorical") 
    
    hist = model.fit_generator(
        train_flow, 
        steps_per_epoch=num_train_imgs//batch_size,
        verbose=2, 
        validation_data=valid_flow,   # to be used later
        validation_steps=num_valid_imgs//batch_size,
        epochs=epochs)

    history = hist.history    
    
    if cnn_epochs > 0:
        model = unfreeze_cnn_layers(model)
        hist_cnn = model.fit_generator(
            train_flow, 
            steps_per_epoch = num_train_imgs//batch_size,
            verbose = 2, 
            validation_data = valid_flow,   # to be used later
            validation_steps = num_valid_imgs//batch_size,
            epochs=cnn_epochs)

        for key in history.keys():
            if type(history[key]) == list:
                history[key].extend(hist_cnn.history[key])

    if save_path is not None:
        model.save(save_path)
    
    return history

def compare_models(num_classes = 10):

    def compare_model(base_mod, num_classes):

        model = create_pretrained_model(num_classes = num_classes, base_mod=base_mod)
        history = train(model, epochs=40, cnn_epochs = 0, num_classes=num_classes, batch_size=16)
        accs = history['val_acc']
        print("accs", accs)
        avg_acc = np.mean(history['val_acc'][-10:])
        MAP, _ = print_model_test_info(model, num_classes)
        return avg_acc, MAP

    base_models = ['InceptionV3', 'Xception', 'ResNet50']
    
    results = []
    for base_model in base_models:
        # avg_acc, MAP = compare_model(base_model)
        avg = 
        results.append((avg_acc, MAP))
    
    save_bar_plot(results, base_models)
        
    return results
    
    #model = create_pretrained_model(num_classes = 3, base_mod='Xception')
    #model = create_pretrained_model(num_classes = 3, base_mod='InceptionV3')
    #model = create_pretrained_model(num_classes = 3, base_mod='MobileNet')
    #model = create_pretrained_model(num_classes = 3, base_mod='InceptionResNetV2'

def main():
    print("Run short training with InceptionV3 and save results.")
    num_classes = 10
    model = create_pretrained_model(num_classes=num_classes)
    histories = train(model, num_classes, epochs=2, cnn_epochs=2)
    print("HISTORIES:")
    print(histories)
    run_name = tools.get_run_name()
    tools.save_learning_curves(histories, run_name)
    csv_path = os.path.join("plots/", run_name, "data.csv")
    ut.write_csv_dict(histories,
                      keys=['loss', 'acc', 'val_loss', 'val_acc'],
                      filename=csv_path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
        exit()
    if "--class-graph" in sys.argv:
        tools.draw_num_classes_graphs()
        exit()
    print("given command line options unknown.")
    
