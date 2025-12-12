# compose the NN model
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dense, Activation, Input
 
from drawing import *
from datagen import generate_linear_data
from tqdm import tqdm
import pandas as pd
import warnings
import click


#############################################################################
###############################  LINEAR REGRESSION  #####################
#############################################################################
def init_regression_model(optimizer='sgd', loss='mse', metrics=['mse'], kernel_initializer='glorot_uniform'):
    """
    Create a one-neuron network to perform linear regression
    Parameter:
        - optimizer [string] : algorithm to optimize the regression parameters
        - loss      [string] : cost function to minimize on the train dataset
        - metrics   [list]   : cost functions to evaluate on the validation dataset
    Return:
        - model     [Keras.Sequential]
    """
    model=tf.keras.Sequential()
    model.add(
        Dense(
            1, input_shape=(1,),
            kernel_initializer=kernel_initializer
        )
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def regression_search(
    n_sigma:int, 
    n_points:int,

    sigma_min=0.,
    sigma_max=1.,

    epochs_min=10,
    epochs_max=100,

    train_min =10,
    train_max=500
):

    warnings.filterwarnings('ignore')  # Suppress all warnings

    
    sigmas=np.random.uniform(sigma_min, sigma_max, n_sigma)

    epochs=np.random.uniform(epochs_min, epochs_max, n_points).astype(int)
    train_slice=np.random.uniform(train_min, train_max, n_points).astype(int)

 

    m=2
    b=1

    df=pd.DataFrame(
        columns=['sigma', 'epochs', 'train_dim', 'loss', 'val_loss']
    )
    for sigma in sigmas:
        for k in tqdm(range(n_points), desc=f'sigma={"%.2f"%sigma}'):
            #print(comb)
            x_train, y_train, x_valid, y_valid, y_target = generate_linear_data(
                m, b, sigma=sigma, seed=0, traindim=train_slice[k] 
            )
            mr = init_regression_model()

            hist=mr.fit(
                x_train, y_train, batch_size=32,
                epochs=epochs[k], shuffle=True,
                validation_data=(x_valid, y_valid),
                verbose=0
            )
            df.loc[len(df)]={
                    'sigma' :sigma, 
                    'epochs':epochs[k],
                    'train_dim':train_slice[k], 
                    'loss':hist.history['loss'][-1], 
                    'val_loss':hist.history['val_loss'][-1]
            }
        


    return df

    

    
#############################################################################
############################  DNN for polynomial fit  #######################
#############################################################################

#@tf.function
def NN(
    n_layer:int=2,
    neurons:list=[3, 2],
    activations=['relu'],
    kernel_initializer='glorot_uniform',
    input_shape=(1,),
    output_shape=1
):
    """
    Generate a Deep Neural network specifing the architecture:
    Parameters:
        - n_layer            [int]      : number of hidden layers
        - neurons            [list(int)]: number of neurons of the corresponding index layer
        - activations        [list(str)]: list of activation functions 
            (if less than the layers the last function will be apllied to the remaining layers)
        - kernel_initializer [str]      : Initializer for the `kernel` weights matrix.
    Return:
        model [Keras.Sequential]
    """

    if (len(neurons)!=n_layer):
        raise ValueError('ERROR: List of neurons incompatible with number of layers')
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape)) #input layer

    activation_index=0
    
    for layer in range(n_layer):
        model.add(
            Dense(
                neurons[layer], 
                activations[activation_index],
                kernel_initializer=kernel_initializer # to confront architectures
            )
        )

        if activation_index+1<len(activations):
            activation_index+=1
    model.add(Dense(output_shape, activation='linear'))

    return model


def NN2(
    n_layer:int=2,
    neurons:list=[3, 2],
    activations=['relu'],
    kernel_initializer='glorot_uniform'
):
    """
    Generate a Deep Neural network specifing the architecture:
    Parameters:
        - n_layer            [int]      : number of hidden layers
        - neurons            [list(int)]: number of neurons of the corresponding index layer
        - activations        [list(str)]: list of activation functions 
            (if less than the layers the last function will be apllied to the remaining layers)
        - kernel_initializer [str]      : Initializer for the `kernel` weights matrix.
    Return:
        model [Keras.Sequential]
    """

    if (len(neurons)!=n_layer):
        raise ValueError('ERROR: List of neurons incompatible with number of layers')
    model = tf.keras.Sequential()
    model.add(Input(shape=(2,))) #input layer

    activation_index=0
    
    for layer in range(n_layer):
        model.add(
            Dense(
                neurons[layer], 
                activations[activation_index],
                kernel_initializer=kernel_initializer # to confront architectures
            )
        )

        if activation_index+1<len(activations):
            activation_index+=1
    model.add(Dense(1, activation='linear'))

    return model

#@click.method()
def test_arch(
    nlayers:int, x_train, y_train, x_valid, y_valid, 
    optimizer:str='sgd', activation:str='relu',
    epochs:int=100, neurons_perlayer:list=[16, 32],
    loss_func:str='mse' 
):
    """
    Test DNN architectures exploring the different combination of 
    neurons per layer on the fixed number of layers

    Parameters:
        - nlayers           [int]
            number of layers of the NN
        - x_train, y_train  [array, array]
            train dataset
        - x_valid, y_valid  [array, array]
            validation dataset
        - optimizer         [str]
            optimizer from the keras.optimizers
        - activation        [str]
            neurons activation function
        - epochs            [int]
            number of epochs for the training process
        - neurons_perlayer  [list[int]]
            list of the possible values of neurons per layer,
            all possible combination of this neurons per layer 
            are explored
        - loss_func         [str]
            loss function to minimize during the training 

    Return:
        - best_loss     [float]
            best validation loss obtained from the tested architectures
        - best_index    [int]
            index corresponding at the best validation loss in the tested combinations
        - combinations  [array[list[int]]]
            contain the different combinations list of neuron per layers
            first index: combination
            second index: layer's number
        - loss          [list[list[float]]]
            contains every architecture loss along epochs
        - val_loss          [list[list[float]]]
            contains every architecture validation loss along epochs            
        - x_predicted   [list[float]]
            list of inputs used to make predictions
        - y_predicted   [list[list[float]]]
            list of predictions made from every tested architectures 
    Example:
        test_arch(
            nlayers=2, x_train, y_train, 
            x_valid, y_valid, 
            neurons_perlayer=[16, 32]
        )
        will test the combination of DNN: 
            [n_neuron-layer1 = 16, n_neuron-layer2 = 16]
            [n_neuron-layer1 = 16, n_neuron-layer2 = 32]
            [n_neuron-layer1 = 32, n_neuron-layer2 = 16]
            [n_neuron-layer1 = 32, n_neuron-layer2 = 32]
    """
    
    x_predicted = np.linspace(-1, 1, 100)
    
    combinations=np.array(
        np.meshgrid(*[neurons_perlayer]*nlayers)
      ).T.reshape(-1, nlayers)
    
    loss=[]
    val_loss=[]
    y_predicted=[]
    npar=[]

    best_vloss=None
    best_index=None

    for index, comb in enumerate(tqdm(combinations)):
        md=NN(nlayers, comb, [activation])
        #md.summary()
    
        md.compile(optimizer=optimizer, loss=loss_func, metrics=[loss_func])
        history=md.fit(
            x=x_train, y=y_train,
            batch_size=32, epochs=epochs,
            shuffle=True, validation_data=(x_valid, y_valid),
            verbose=0
        )
    
        if best_vloss==None:
            best_vloss=min(history.history['val_loss'])
            best_index=index
        elif min(history.history['val_loss'])<best_vloss:
            best_vloss=min(history.history['val_loss'])
            best_index=index

        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        ypred=md.predict(x_predicted, verbose=0)
        y_predicted.append(ypred)
        npar.append(md.count_params())
 
    return  best_vloss, best_index, combinations, loss, val_loss, npar, x_predicted, y_predicted


