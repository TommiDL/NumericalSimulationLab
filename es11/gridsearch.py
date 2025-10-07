from model_def import *
from datagen import *

import pandas as pd
import os

def print_optimizers():
    print("""
Optimizer list:
    SGD
    RMSprop
    Adam
    AdamW
    Adadelta
    Adagrad
    Adamax
    Adafactor
    Nadam
    Ftrl
    Lion
    Lamb
    Loss Scale Optimizer
    Muon


    """)

def print_losses():
    print("""
Loss functions list: 
    -binary_crossentropy: 
    -binary_focal_crossentropy: 
    -categorical_crossentropy: 
    -categorical_focal_crossentropy: 
    -categorical_hinge: 
    -cosine_similarity: 
    -ctc: [Connectionist Temporal Classification] 
    -deserialize:   
        Deserializes a serialized loss class/function instance.
    -dice: 
    -hinge:     
    -huber:     
    -kld: [Computes Kullback-Leibler divergence] 
    -logcosh: 
        Logarithm of the hyperbolic cosine of the prediction error.
    -mae: [mean absolute error]
    -mape:[mean absolute percentage error]
    -mse: [mean squared error]
    -msle: [mean squared logarithmic error] 
    -poisson: 
    -serialize: 
        Serializes loss function or Loss instance.
    -sparse_categorical_crossentropy: 
    -squared_hinge: 
    -tversky: 
    """)


def main():

    print_optimizers()
    optimizer=   input("optimizer >> ")

    activation = input("activation >> ")

    print_losses()  
    loss_func =  input("loss >> ")

    # create dirs structure 
    os.makedirs(optimizer+"/"+activation, exist_ok=True)
    
    # max number of layers
    Nmax=4
    Nmin=1

    # dataset to store results
    #   loss 
    bl=pd.DataFrame(
        columns=[f'loss_{k}layers' for k in range(1, Nmax)]+[f'val_loss_{k}layers' for k in range(1, Nmax)]
    )
    #   predictions
    bpred=pd.DataFrame(columns=[f'xpred_{k}layers' for k in range(1, Nmax)]+[f'ypred_{k}layers' for k in range(1, Nmax)])
    bpar=pd.DataFrame(
        columns=[f'par_{k}layers' for k in range(1, Nmax)]
    )


    # generate polynomial datas    
    x_train, y_train, x_valid, y_valid, y_target = generate_poly_data(pol)
    
    combfile = ""
    lossfile = ""
    predfile = ""

    if loss_func=="mse":
        combfile = "combinations.dat"
        lossfile = "losses.dat"
        predfile = "predictions.dat"
        parafile = "parameters.dat"
    else:
        combfile = "combinations"+loss_func+".dat"
        lossfile = "losses"+loss_func+".dat"
        predfile = "predictions"+loss_func+".dat"
        parafile = "parameters"+loss_func+".dat"

    # open file to write combinations of neuron per layer
    with open(optimizer+"/"+activation+"/"+combfile, "w") as myfile:
        myfile.write("best_combinations\n")


    for k in range(Nmin, Nmax):

        bestloss, bestindex, comb, loss, val_loss, npar, x_pred, y_pred = test_arch(
            k, x_train, y_train,x_valid, 
            y_valid,neurons_perlayer=[2, 6, 12, 32], 
            optimizer=optimizer, activation=activation, 
            loss_func=loss_func

        )
        
        # update dataframes
        bl[f'loss_{k}layers']=loss[bestindex]
        bl[f'val_loss_{k}layers']=val_loss[bestindex]

        bpred[f'xpred_{k}layers']=x_pred
        bpred[f'ypred_{k}layers']=y_pred[bestindex]

        bpar[f'par_{k}layers']=npar[bestindex]

        # save on files
        bl.to_csv(optimizer+"/"+activation+"/"+lossfile)
        bpred.to_csv(optimizer+"/"+activation+"/"+predfile)
        bpar.to_csv(optimizer+"/"+activation+"/"+parafile)
        with open(optimizer+"/"+activation+"/"+combfile, "a") as myfile:
            myfile.write(" ".join(map( str, comb[bestindex])))
            myfile.write("\n")


        #plt.show()
if __name__=='__main__':
    main()