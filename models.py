# racing models
import numpy as np
# from copy import deepcopy
# import time
# import warnings
import keras.backend as K
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM, concatenate, Masking
# from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, Callback
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn import linear_model
# from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

import tools as t

            
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


def _get_callbacks(earlystop):

    callbacks = []
    if earlystop>0:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.00001,  
                                       patience=earlystop,
                                       # patience=np.amin([np.amax([epochs/10, 5]),75]), 
                                       verbose=1, mode='auto'))
        # save weights at best iteration, and restore at end of training
        # reset = True --> takes best iteration for each CV-fold
        callbacks.append(GetBest(monitor='val_loss', verbose=1, mode='auto', reset = True))
        print("evaluating with early stopping")

    return callbacks


# create XGB model 
def xgb_model(cfg):
    
    model = xgb.XGBRegressor(n_estimators=cfg['n_estimators'], 
                                 learning_rate = cfg['lr'], 
                                 gamma = cfg['gamma'], 
                                 subsample = cfg['subsample'], 
                                 colsample_bytree = cfg['cols_bt'], 
                                 max_depth = cfg['maxdepth'])
    return model
      

# passing a dictionary (cfg) to mlp triggers a deprecation warning - passing simple parameters does not (??)
def mlp(cfg, input_dim, dropout=False, L1L2=False):

    # print("create mlp using learning rate:", lr)
    
    if L1L2==True:
        kernel_regularizer=l1_l2(l1=cfg['l1'], l2=cfg['l2'])
        print("create mlp using L1L2 regularisation")
    else: 
        kernel_regularizer = None

    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))
    if dropout==True:
        model.add(Dropout(0.2))
        print("create mlp using Dropout")
    model.add(Dense(512, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))
    if dropout==True:
        model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))        
    if dropout==True:
        model.add(Dropout(0.8))
    
    model.add(Dense(1, kernel_initializer='normal'))
    
    opt = optimizers.SGD(lr=cfg['lr'])
    
    model.compile(loss='mean_squared_error', optimizer=opt)  # no adaptive learning rate (--> later compare to exponential)
    
    return model


def lstm(param_dim, segment_dim, lr=0.01):   # keras default is lr=0.001, but runs better at 0.01
    
    # lstm branch of model, using masking for random values
    lcs_input = Input(shape=(None, segment_dim), name='lcs_input')
    # segment array filled with zeros as mask
    
    # masking = Masking(mask_value=[0,0,0,0,0,0])(lcs_input)    # input (timesteps / features)
    masking = Masking(mask_value=np.zeros(segment_dim))(lcs_input)    # input (timesteps / features)

    lstm_next = LSTM(64, return_sequences=True)(masking)
    # lstm_next = LSTM(128, return_sequences=True)(lstm_next)    
    lstm_out = LSTM(64)(lstm_next)

    # branch for non sequential configuration data
    cfgs_input = Input(shape=(param_dim,), name='cfgs_input')
    
    x = concatenate([cfgs_input, lstm_out])    
    x = Dense(64)(x)
    x = Dense(64)(x)
    main_output = Dense(1, name='main_output')(x)
    
    model = Model(inputs=[cfgs_input, lcs_input], outputs=[main_output])

    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse'])
    return model


# ToDo: pass learning rate at creation of model
def train_mlp(model, configs, Y, cfg, split, epochs):
    
    configs_trn, configs_val = configs[:split], configs[split:]
    y_trn, y_val = Y[:split], Y[split:]
    # print(configs_trn.shape, configs_val.shape, y_trn.shape, y_val.shape)    

    hist = model.fit(configs_trn, y_trn,
                     batch_size=cfg['batch_size'], 
                     epochs=epochs,
                     validation_data=(configs_val, y_val))
    return hist


def _pred_mlp(model, X, idx, batch_size):

    configs = X[idx]
    y_pred = model.predict(configs, batch_size=batch_size)
    y_pred = y_pred.reshape(idx.shape[0])    
    
    return y_pred


def eval_mlp(model, X, Y, split, batch_size):
    
    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx_trn = np.arange(0,split)
    idx_val = np.arange(split,sample_no)     
    
    y_pred_trn = _pred_mlp(model, X, idx_trn, batch_size)    
    y_pred_val = _pred_mlp(model, X, idx_val, batch_size)

    mse_trn = ((y_pred_trn - Y[idx_trn]) ** 2).mean()
    mse_val = ((y_pred_val - Y[idx_val]) ** 2).mean()
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_trn, mse_val))     
    
    return mse_trn, mse_val    


# mode "nextstep": train on predicting last point of observed points (e.g. 5 of 5)
# mode "finalstep": train on predicting final point of learning curve (#40)
# masking: LSTM can process sequences of arbitrary lenght but not train on sequences of different length within one 
# batch_array or generator. To do so requires bringing all sequences physically to the same lenght (e.g.40), 
# and "mask" last values with zeros (e.g. 15 zeros if actual sample has lenght 25)
def _train_lstm(model, X, Y, trn_idx, val_idx, 
                batch_size, epochs, callbacks = None, verbose = 0):
    
    # segments is 3d array [batch_size, segement_arr, seg_features] where segment_arr is 
    # zero padded list of segments with len of longest segment_list of whole data_set 
    # shorten the array to len of longest segment_list in current batch
    def shorten_zeros(segments):
        max_len = 0        
        for line in segments:        
            j = 0
            eol = False
            while not eol:
                j=j+1
                if j>=line.shape[0]:
                    eol = True
                elif np.all(line[j] == 0):
                    eol = True
            # print("this_len", j)        
            max_len = max(max_len, j)     
        return segments[:,:max_len]
    
    # generate input for fit_generator()
    def generate_seqs(X, Y, idx, batch_size):

        while 1:
            params   = X[0][idx]
            segments = X[1][idx]
            y        = Y[idx]
            # x, y = truncate_lcs(lcs, steps)

            for i in range(0, y.shape[0], batch_size):
                
                segs_batch_short = shorten_zeros(segments[i : i+batch_size])
                # print("this batch shortens segments to len: ", segs_batch_short.shape[1])
                # if segs_batch_short.shape[1] > 800:
                #     print(" !!!!! fucking error !!!")
                
                yield ([params[i : i+batch_size], segs_batch_short], y[i : i+batch_size])      
                   
    print("train lstm")
        

    trn_generator = generate_seqs(X, Y, trn_idx, batch_size)        
    val_generator = generate_seqs(X, Y, val_idx, batch_size)        
        
    hist = model.fit_generator(trn_generator,
                               steps_per_epoch = int(np.ceil(trn_idx.shape[0] / batch_size)), 
                               epochs=epochs, 
                               callbacks=callbacks,
                               validation_data = val_generator,
                               validation_steps = int(np.ceil(val_idx.shape[0] / batch_size)), 
                               verbose = verbose)
    return hist


# taking train/valid split point as parameter for manual experiments
# steps tuple (training-timesteps , validation-timesteps), train-steps == 0 --> random length
def train_lstm(model, X, Y, split=200, batch_size=20, epochs=3, verbose = 0, earlystop = 0):

    sample_no = X[0].shape[0]
    trn_idx, val_idx = np.arange(0,split), np.arange(split,sample_no)
    return _train_lstm(model, X, Y, trn_idx, val_idx, batch_size, epochs, 
                       callbacks=_get_callbacks(earlystop), verbose=verbose)
        
# X is tuple (params,sections)
def _pred_lstm_direct(model, X, steps, idx, batch_size):

    print("evaluate lstm with consideration of configs")
    configs, lcs = X[0], X[1]
    lcs_val   = lcs[idx][:,:steps]
    X_val     = [configs[idx], lcs_val]     
    Y_val     = lcs[idx][:,-1]  

    y_pred = model.predict(X_val, batch_size=batch_size) 
    y_pred = y_pred.reshape(idx.shape[0])
    
    return y_pred

       
def eval_lstm_direct(model, X, Y, steps, split, batch_size):

    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx_trn = np.arange(0,split)
    idx_val = np.arange(split,sample_no)    

    y_pred_trn = _pred_lstm_direct(model, X, steps, idx_trn, batch_size)
    y_pred_val = _pred_lstm_direct(model, X, steps, idx_val, batch_size)
    
    mse_trn = ((y_pred_trn - Y[idx_trn]) ** 2).mean()
    mse_val = ((y_pred_val - Y[idx_val]) ** 2).mean()
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_trn, mse_val))  
    
    return mse_trn, mse_val

   

# create and train model and evaluate by cross validation
# steps = (training steps, lsit of validation steps) e.g. (10,[5,10,20,30])
def eval_cv(model_type, X, Y, steps=(0,[0]), cfg={}, epochs=0, splits=3, 
            earlystop=0, dropout=False, L1L2=False):
    
    print("cross validate {} epochs, train on {} steps, validate on {} steps".format(epochs, steps[0], steps[1]))
    print("config {}".format(cfg))

    
    callbacks = _get_callbacks(earlystop)

    if model_type == 'ridge':
        model = linear_model.Ridge(alpha = cfg['alpha'])
    elif model_type == 'xgb':
        model = xgb_model(cfg)
    elif model_type == 'mlp':
        model = mlp(cfg, dropout=dropout, L1L2=L1L2)
    elif model_type == 'lstm':
        model = lstm(cfg['lr'])
    else:
        print("invalid model type", model_type)

    if model_type in ['lstm','mlp']:
        init_weights = model.get_weights()

    results_val=[]    # list of validation results for each fold
    results_trn=[]    # ... but for task with 'nextstep' also on training data
    # y_pred = np.zeros(Y.shape[0])  # successively stores predictions on validation folds (on all unseen data)
    
    #val_preds = []
    #for i in range(steps[1]):
    #    val_preds.append(np.zeros(y.shape[0]))
    y_preds = [np.zeros(Y.shape[0]) for i in range(len(steps[1]))]  # for returning all predictions 
        
    fold_count = 0
    kfold = KFold(n_splits=splits, random_state=t.seed)
    for trn_idx, val_idx in kfold.split(Y):
        # trn_true, val_true = Y[trn_idx], Y[val_idx]
        fold_count += 1
        # faster solution: train once, evaluate over all cases [5,10,20,30]
        if model_type in ['lstm','mlp']:            
            model.set_weights(init_weights)     # to make results reproducible always start with same init weights
        print("train fold {} on {} steps, validation on {} steps".format(fold_count, steps[0], steps[0]))
        # steps[1] for validation here only relevant as criterion for early stopping
        if model_type in ['xgb','ridge']:
            model.fit(X[trn_idx], Y[trn_idx])
        elif model_type == 'mlp':
            model.fit(X[trn_idx], Y[trn_idx],
                      batch_size=cfg['batch_size'], epochs=epochs, callbacks=callbacks,
                      verbose = 0, validation_data=(X[val_idx], Y[val_idx]))                
        elif model_type in ['lstm']:
            _train_lstm(model, X, steps=(steps[0], steps[0]), idx=(trn_idx, val_idx), 
                        batch_size=cfg['batch_size'], epochs=epochs, 
                        callbacks=callbacks, mode=mode)

        # now model has weights of best run during last training (based on val_loss)
        # now evaluate on train and valid data of given steps [5,10,20,30]
        trn_mses, val_mses = [],[]    # list of mses of folds
        trn_pred, val_pred = [],[]    # list of predictions of one fold
        trn_true = Y[trn_idx].reshape(trn_idx.shape[0])
        val_true = Y[val_idx].reshape(val_idx.shape[0])            
        
        if model_type in ['ridge', 'xgb']:
            trn_pred = model.predict(X[trn_idx]).reshape(trn_idx.shape[0])
            val_pred = model.predict(X[val_idx]).reshape(val_idx.shape[0])
            y_preds[0][val_idx]=val_pred
            trn_mses.append(((trn_pred - trn_true) ** 2).mean())
            val_mses.append(((val_pred - val_true) ** 2).mean())                
        elif model_type == 'mlp':
            trn_pred = _pred_mlp(model, X, trn_idx, batch_size=cfg['batch_size'])                
            val_pred = _pred_mlp(model, X, val_idx, batch_size=cfg['batch_size'])
            y_preds[0][val_idx]=val_pred
            trn_mses.append(((trn_pred - trn_true) ** 2).mean())
            val_mses.append(((val_pred - val_true) ** 2).mean())                
        else:   # if lstm or xgb_next, list of validation data
            for i, val_steps in enumerate(steps[1]):
                val_pred = _pred_lstm_direct(model, X, val_steps, val_idx, 
                                             cfg['batch_size'])
                trn_pred = _pred_lstm_direct(model, X, val_steps, trn_idx,
                                             cfg['batch_size'])

                y_preds[i][val_idx]=val_pred                         

                trn_mses.append(((trn_pred - trn_true) ** 2).mean())
                val_mses.append(((val_pred - val_true) ** 2).mean())
                # print("Y.shape after", Y.shape)

                print("validate on {} steps, mse on train / validation data: {:.5f} / {:.5f}"\
                      .format(val_steps, trn_mses[-1], val_mses[-1]))

        # y_pred[val_idx] = val_pred   # store results only for last value in list of val_steps

        results_val.append(val_mses)
        results_trn.append(trn_mses)

    results_val, results_trn = np.array(results_val), np.array(results_trn)
    val_means, trn_means = [], []
    
    val_means = abs(np.round(results_val.mean(axis=0),5))
    print("MSE on validation data on {} steps: means over folds: *** {} ***".format(steps[1], val_means))
    print("Results validation data of all Folds: \n{}".format(np.round(results_val,5)))
    
    if results_trn.shape[0] > 0:
        trn_means = abs(np.round(results_trn.mean(axis=0),5))
        print("MSE on train data on {} steps: means over folds: *** {} ***".format(steps[1], trn_means))
        print("Results training data of all Folds: \n{}".format(np.round(results_trn,5)))
        
    mse_total = ((y_preds[0] - Y.reshape(Y.shape[0])) ** 2).mean()
    print("mse over all validation data", mse_total)

    result = {'y_preds'   : y_preds,
              'mse'       : mse_total, 
              'trn_means' : trn_means, 
              'val_means' : val_means}
        
    return result  
    # return y_pred, mse_total, trn_means, val_means
