# Deep learning lab course final project.
# Kaggle whale classification.

import os
import sys
import numpy as np
import time
import Augmentor

#import h5py
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping, Callback

import utilities as ut
import keras_tools as tools

# Modify Augmentor Pipeline class, write keras_generator which samples without replacement
class MyPipeline(Augmentor.Pipeline):
    
    def __init__(self, source_directory=None, output_directory="output", save_format=None):
        super(MyPipeline, self).__init__(source_directory, output_directory, save_format)
        

    def my_keras_generator(self, batch_size, scaled=True, image_data_format="channels_last", with_replace=True):

        if image_data_format not in ["channels_first", "channels_last"]:
            warnings.warn("To work with Keras, must be one of channels_first or channels_last.")

        while True:

            X = []
            y = []

            for i in range(batch_size):
                if with_replace:
                    # standard version: Select random image, get image array and label
                    image_index = random.randint(0, len(self.augmentor_images)-1)
                else:
                    # My version: select images one by one --> to be implemented
                    # likely augmentor_images contains all images (test), so run whole array batch wise (in endless "while" loop)
                    # image_index = gaga
                    print("not implemented")
                    
                numpy_array = np.asarray(self._execute(self.augmentor_images[image_index], save_to_disk=False))
                label = self.augmentor_images[image_index].categorical_label

                # Reshape
                w = numpy_array.shape[0]
                h = numpy_array.shape[1]

                if np.ndim(numpy_array) == 2:
                    l = 1
                else:
                    l = np.shape(numpy_array)[2]

                if image_data_format == "channels_last":
                    numpy_array = numpy_array.reshape(w, h, l)
                elif image_data_format == "channels_first":
                    numpy_array = numpy_array.reshape(l, w, h)

                X.append(numpy_array)
                y.append(label)

            X = np.asarray(X)
            y = np.asarray(y)

            if scaled:
                X = X.astype('float32')
                X /= 255

            yield (X, y)


# stores weights of best values and restore these after training --> not done by default by earlystop !! 
# modified keras ModelCheckpoint class see https://github.com/keras-team/keras/issues/2768 code user louis925
# enhanced with "reset" parameter
class GetBest(Callback):
    """Get the best model at the end of training.
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    # Example
        callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
        mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """
    # reset best found value at beginning of each new training
    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1, reset = True):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.reset = reset
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                # print("choose max as mode")
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                # print("choose min as mode")                
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        if self.reset == True:      # useful if multiple calls of fit(), e.g. during cross validation 
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s is %0.5f, did not improve' %
                              (epoch + 1, self.monitor, current))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)
            
        

def get_run_name(prefix="run", additional=""):
    return "_".join([prefix, 
                     datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                     additional])




# Use pretrained model as described in https://keras.io/applications/

def _create_pretrained_model(config_dict, num_classes):
    #
    # extract relevant parts of configuration
    #
    num_dense_units_list = []
    dropout_list = []
    base_model = config_dict['base_model']
    num_dense_layers = config_dict['num_dense_layers']
    for i in range(num_dense_layers):
        num_dense_units_list.append(config_dict['num_dense_units_' + str(i)])
    activation = config_dict['activation']
    dropout = config_dict["dropout"]
    if dropout==True:
        for i in range(num_dense_layers):
            dropout_list.append(config_dict['dropout_' + str(i)])
    optimizer = config_dict['optimizer']
    learning_rate = config_dict['learning_rate']

    #
    # load pre-trained model
    #
    if base_model == 'InceptionV3':
        pretrained_model = InceptionV3(weights='imagenet', include_top=False)  
    elif base_model == 'Xception':
        pretrained_model = Xception(weights='imagenet', include_top=False)
    elif base_model == 'ResNet50':
        pretrained_model = ResNet50(weights='imagenet', include_top=False)
    elif base_model == 'MobileNet':
        pretrained_model = MobileNet(weights='imagenet', input_shape=(224, 224,3), include_top=False)
    elif base_model == 'InceptionResNetV2':
        pretrained_model = InceptionResNetV2(weights='imagenet', include_top=False)
    else:
        print("invalid model: ", base_model)
    
    x = pretrained_model.output

    # for i, layer in enumerate(pretrained_model.layers):
    #    print(i, layer.name)    
    
    
    #
    # add fully connected layers
    #
    x = pretrained_model.output

    x = GlobalAveragePooling2D()(x)
    for i in range(num_dense_layers):
        x = Dense(num_dense_units_list[i], activation=activation)(x)
        if dropout==True:
            x = Dropout(dropout_list[i])(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # finish building combined model, lock parameters of pretrained part
    #
    model = Model(inputs=pretrained_model.input, outputs=predictions)
    for layer in pretrained_model.layers:
        layer.trainable = False
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=learning_rate)
    elif optimizer == 'Adam':
        opt = optimizers.Adam(lr=learning_rate)
    elif optimizer == 'RMSProp':
        opt = optimizers.RMSprop(lr=learning_rate)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))
    # compile the model (should be done *after* setting layers to
    # non-trainable)
    # metrics='accuracy' causes the model to store and report accuracy (train
    # and validate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.name = base_model    # to identify model in "unfreeze_layers() and "train()" function
    # print("successfuly created model: ", model.name)    
    
    return model


# "unfreeze_percentage" is fraction of whole CNN model to be unfrozen
# range 0.0 up to 0.3 - Values above 0.3 are interpreted as 0.3
def _unfreeze_cnn_layers(model, config_dict, unfreeze_percentage):
    # we chose to train the top X layers, where X is one of the nodes of the CNN
    # the first 2 layer blocks and unfreeze the rest:
    # to visualize structure call tools.visualize_model(model=model, filename="MobileNet_visualization.png",show_shapes=False)
    
    # unfreeze_percentage = config_dict["unfreeze_percentage"]
    # unfreeze_percentage = min(unfreeze_percentage, 0.4)  
    # unfreeze_percentage = max(unfreeze_percentage, 0.0)  
    unfreeze_blocks = 0
    if model.name == 'InceptionV3':
        top_nodes = [280, 249, 229, 197, 165, 133, 101, 87, 64, 41, 0]   # nodes of top layer-blocks: possible cut_off points
        unfreeze_blocks = int(11 * unfreeze_percentage)  # 
    elif model.name == 'Xception':
        top_nodes = [126, 116, 106, 96, 86, 76, 66, 56, 46, 36, 26, 16, 0]   
        unfreeze_blocks = int(13 * unfreeze_percentage)  # 
    elif model.name == 'ResNet50':
        top_nodes = [161, 152, 140, 130, 120, 110]   
        unfreeze_blocks = int(16 * unfreeze_percentage)  # 
    elif model.name == 'MobileNet':    # no nodes, cut_off after activations
        top_nodes = [79, 76, 73, 70, 67, 64, 61, 58, 55, 52, 49, 46, 43, 40, 37,34,32,28,25,22,19,16,13,10,7,4,0]          
        unfreeze_blocks = int(27 * unfreeze_percentage)  # 
    elif model.name == 'InceptionResNetV2':    # no nodes, cut_off after activations
        top_nodes = [761, 745, 729, 713, 697, 681, 665, 649, 633, 618, 594, 578, 562, 546]
        unfreeze_blocks = int(43 * unfreeze_percentage)  # 
              
    else:
        print("invalid model: ", model.name)

    if unfreeze_blocks > 0:
        
        cut_off = top_nodes[unfreeze_blocks-1]                
        for layer in model.layers[:cut_off]:
           layer.trainable = False
        for layer in model.layers[cut_off:]:
           layer.trainable = True        

        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=config_dict['cnn_learning_rate'], momentum=0.9), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print("\n ****** {} unfrozen {} top blocks, cut_off after layer {} ******".format(model.name, unfreeze_blocks, cut_off-1))
    else:
        print("\n ****** {} no layers unfrozen".format(model.name))
        
    return model


def get_callbacks(earlystop):   # -1 --> Earlystop = True
    
    callbacks = []
    if earlystop:    
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.00001,                                        
                                       # patience=np.amin([np.amax([epochs/10, 5]),75]), 
                                       patience=20,
                                       verbose=1, mode='auto'))
        # save weights at best iteration, and restore at end of training
        # reset = True --> takes best iteration for each CV-fold
        callbacks.append(GetBest(monitor='val_loss', verbose=1, mode='auto', reset = True))
        print("evaluating with early stopping")

    return callbacks

def train(config_dict, 
          epochs,
          model=None,
          num_classes=10,
          save_model_path="model/kaggle_model",  
          save_data_path="plots",
          train_dir="data/model_train",
          train_csv="data/model_train.csv",
          valid_dir="data/model_valid",
          valid_csv="data/model_valid.csv",          
          train_valid_split=0.9,
          earlystop = False,
          augmented = 'keras', 
          create_case = True,
          unfreezings = []):   # gaga

    global num_train_imgs, num_valid_imgs   # notlösung, da nich weiß, wie static local variables.... 
    
    start_time = time.time()
    #
    # extract relevant parts of configuration
    #
    # cnn_unlock_epoch = config_dict["cnn_unlock_epoch"]
    # unfreeze_percentage = config_dict["unfreeze_percentage"]
    batch_size = config_dict['batch_size']
    
    #
    # get model to train, determine training times
    #
    if model is None:
        model = _create_pretrained_model(config_dict, num_classes)
    
    if model.name == 'InceptionV3' or model.name == 'Xception' or model.name == 'InceptionResNetV2':
        target_size = (299, 299)
    elif model.name == 'ResNet50' or model.name == 'MobileNet':
        target_size = (224, 224)
    else:
        print("invalid model: ", model.name)
    print("training model", model.name)    

    #
    # prepare training data
    #
    
    # create environment on filesystem with new random train/valid split
    if create_case:
        num_train_imgs, num_valid_imgs = ut.create_small_case(
            sel_whales=np.arange(1, num_classes+1),
            train_dir=train_dir,
            train_csv = train_csv,
            valid_dir=valid_dir,
            valid_csv = valid_csv,
            train_valid=train_valid_split,
            sub_dirs=True)
    
    if augmented == 'keras':
        # old version using Keras built in replaced by augmentor stuff 11.04.2018
        train_gen = image.ImageDataGenerator(    # ImageDataGenerator class initialisation
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

        # image-samples are in subdirectories of train_dir, names of these subdirectories are class names

        train_flow = train_gen.flow_from_directory(  # return 
            train_dir,
            # save_to_dir="data/model_train/augmented",    
            # color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size, 
            class_mode="categorical")
    elif augmented == 'augmentor':
        p = MyPipeline(train_dir, output_directory='')    # Augmentor Pipeline, modified by me
        p.resize(1, target_size[0], target_size[1], resample_filter='NEAREST')
        p.greyscale(1)
        p.rotate(probability=0.75, max_left_rotation=25, max_right_rotation=25)
        p.zoom(probability=0.9, min_factor=0.75, max_factor=1.1)
        p.crop_random(probability=1, percentage_area=0.9)
        p.skew(1, magnitude=0.5)
        train_flow = p.keras_generator(batch_size=batch_size, scaled=True, with_replace=True)  # scaled rescales [0-255] --> [0,1]
    else:    
        train_gen = image.ImageDataGenerator(rescale=1./255,fill_mode="nearest")            
        train_flow = train_gen.flow_from_directory(
            train_dir,target_size=target_size,batch_size=batch_size,class_mode="categorical")        
    
    if valid_dir != None:
        valid_gen = image.ImageDataGenerator(
            rescale=1./255,
            fill_mode="nearest")

        valid_flow = valid_gen.flow_from_directory(
            valid_dir,
            target_size=target_size,
            class_mode="categorical") 

    
    #
    # train fully connected part
    # 
    

    callbacks = get_callbacks(earlystop)
    
    if valid_dir != None:
        hist_dense = model.fit_generator(
            train_flow, 
            steps_per_epoch=num_train_imgs//batch_size,
            verbose=2, 
            callbacks=callbacks,
            validation_data=valid_flow,
            validation_steps=num_valid_imgs//batch_size,
            epochs=epochs)
    else:
        hist_dense = model.fit_generator(
            train_flow, 
            steps_per_epoch=num_train_imgs//batch_size,
            verbose=2,
            callbacks=callbacks,
            epochs=epochs)
    histories = hist_dense.history
    #
    # train the whole model with parts of the cnn unlocked (fixed optimizer!)
    #

    '''
    if len(unfreezings) > 0:
        if epochs <= cnn_unlock_epoch:
            training_epochs_dense = epochs
            training_epochs_wholemodel = 0
        else:
            training_epochs_dense = cnn_unlock_epoch
            training_epochs_wholemodel = epochs - cnn_unlock_epoch    
    '''
    # unfreezings array of tuples (epochs,unfreeze_percentage) e.g. [(20,0.2),(-1,0.4)] where -1 stands for EarlyStop
    for unfreezing in unfreezings:   
    # if training_epochs_wholemodel > 0:
        
        callbacks = get_callbacks(unfreezing[0] == -1)   # -1 --> Earlystop = True
        if (unfreezing[0] == -1):
            epochs = 1000
        else:
            epochs = unfreezing[0]
    
        model = _unfreeze_cnn_layers(model, config_dict, unfreezing[1])
        if valid_dir != None:       
            hist_wholemodel = model.fit_generator(
                train_flow, 
                steps_per_epoch = num_train_imgs//batch_size,
                verbose = 2,
                callbacks=callbacks,
                validation_data = valid_flow,
                validation_steps = num_valid_imgs//batch_size,
                epochs=epochs)
        else:
            hist_wholemodel = model.fit_generator(
                train_flow, 
                steps_per_epoch = num_train_imgs//batch_size,
                verbose = 2,
                callbacks=callbacks,
                epochs=epochs)            
        # concatenate training history
        for key in histories.keys():
            if type(histories[key]) == list:
                histories[key].extend(hist_wholemodel.history[key])
    
    #
    # do final cleanup
    #
    if save_model_path is not None:
        model.save(save_model_path)
        print("model saved: ", save_model_path)

    # if save_data_path is not None:
        # run_name = tools.get_run_name()
        # TODO find bugs and reactivate following lines
        #tools.save_learning_curves(histories, run_name, base_path=save_data_path)
        #csv_path = os.path.join(save_data_path, run_name, run_name + ".csv")
        #ut.write_csv_dict(histories,
        #                  keys=['loss', 'acc', 'val_loss', 'val_acc'],
        #                  filename=csv_path)

    
    # hpbandster_loss = 1.0 - histories['val_acc'][-1]   gaga
    runtime = time.time() - start_time
    # return (hpbandster_loss, runtime, histories)
    return 0, runtime, histories, model


def train_bagging(config_dict, 
          epochs,
          model=None,
          num_classes=10,
          # save_model_path="model/kaggle_model",  
          # save_data_path="plots",
          train_dir="data/model_train",
          train_csv="data/model_train.csv",
          valid_dir="data/model_valid",
          valid_csv="data/model_valid.csv",          
          train_valid_split=0.9,
          earlystop = False,
          augmented = 'keras', 
          bagging_no = 3,     # how many rounds of bagging
          create_case = True, 
          unfreezings = []):

    models = []
    for i in range(bagging_no):
        
        # create new case, with replacement inside classes
        global num_train_imgs, num_valid_imgs   # notlösung, da nich weiß, wie static local variables....     
        num_train_imgs, num_valid_imgs = ut.create_small_case(
            sel_whales=np.arange(1, num_classes+1),
            train_dir=train_dir,
            train_csv = train_csv,
            valid_dir=valid_dir,
            valid_csv = valid_csv,
            train_valid=train_valid_split,
            sub_dirs=True,
            bag_no=i+1)    # count bag_no from 1,2,3,....
        
        _, _, _, model = train(config_dict = config_dict, 
                  epochs = epochs,
                  model=model,
                  num_classes=num_classes,
                  # save_model_path=save_model_path+'_bag'+str(i),  # name models individually
                  save_model_path=None,  # models will be saved as List below via pickle

                  # save_data_path=save_data_path,
                  train_dir=train_dir,
                  train_csv=train_csv,
                  valid_dir=train_csv,
                  valid_csv= valid_csv,          
                  train_valid_split=None,
                  earlystop = earlystop,
                  augmented = augmented, 
                  create_case = False,
                  unfreezings = unfreezings)

        models.append(model)
        
        # pickle_to_file(models, save_model_path)
        
    return models
    
    