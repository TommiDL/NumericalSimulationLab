from model_def import *
from datagen import *

import pandas as pd
import os
from matplotlib.gridspec import GridSpec

import click


@click.group()
def cli():
    pass

def fitsin(neurs:list, opt:str, epochs:int=100, sigma:float = 0.):    

    (x_train, y_train, z_train), (x_valid, y_valid, (z_valid, z_target)) = gen_sinxy(sigma=sigma)
    
    Xtrain = np.column_stack((x_train, y_train))
    Xvalid = np.column_stack((x_valid, y_valid))

    mod = NN2(len(neurs), neurs)
    
    
    #mod.summary()
    
    mod.compile(optimizer=opt, loss='mse', metrics=['mse'])
    
    
    hist = mod.fit(
        x=Xtrain, y=z_train,
        batch_size=32, epochs=epochs, shuffle=True,
        validation_data=(Xvalid, z_valid), verbose = 0
    )
    

    return mod, hist

def plotperf(archs, figname='comparesin.png'):
    
    fig = plt.figure(layout="constrained", figsize=(10, 40))
    #figs = fig.subfigures(len(archs), 2)
    gs = GridSpec(3*len(archs), 3, figure=fig)
    counter=0
    for key, el in archs.items():
        sigmas = el.keys()

        
    
        title = 'NN 2' 
        
        #figs[counter].suptitle(title, x=0.3)
        ax1 = fig.add_subplot(gs[3*counter:3*(counter+1), 0])
        DrawNN([2]+np.array(key.split('x'), dtype=int).tolist()+[1]).draw(ax1)
        ax1.set_title('NN 2x'+key+'x1')
        ax2 = fig.add_subplot(gs[3*counter, 1:])
        ax2.set_title('MSE')

        #ax3 = fig.add_subplot(gs[1:, 1], box_aspect=1)
        #ax3.set_title('validation', y=1)
        
        ax4 = [fig.add_subplot(gs[3*counter+k//2+1, k%2+1], box_aspect=1) for k, sigma in enumerate(sigmas)]
        [ax4[k].set_title(r'$\sigma  = $'+f'{s}') for k,s  in enumerate(sigmas)]
        

        x_pred = np.random.uniform(-3./2., 3./2., 10000)
        y_pred = np.random.uniform(-3./2., 3./2., 10000)
        
        
        for k, sigma in enumerate(sigmas):
            mod = el[sigma]['mod']
            hist = el[sigma]['hist']
    
            
        
            # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
            #ax2.plot(hist.history['loss'], label='training')
            ax2.plot(hist.history['val_loss'], label=f'sigma = {sigma}')
            
            #hx3 = ax3.hexbin(x_valid, y_valid, C=z_valid, vmin=-1.0, vmax=1.0, gridsize=50)
            #plt.colorbar(hx, ax=ax3)
            
            hx4 = ax4[k].hexbin(
                x_pred, y_pred, 
                C=mod.predict(np.column_stack((x_pred, y_pred)), verbose=0), 
                vmin=-1.0, vmax=1.0, gridsize=50)
        counter+=1
        
        ax2.grid()
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('loss')
        ax2.legend(loc=(1.1,0))
        plt.colorbar(hx4, ax=ax4)
        
    
    plt.savefig(figname)


opt_help:str="""Optimizer list:
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


    """
loss_help:str = """Loss functions list: 
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
    """



@click.command(
    'fit_pol',
    help='test different NN configuration to return the best validation loss'
)
@click.option( 
    '-opt', '--optimizer', type=str, help=opt_help, default='sgd'
)
@click.option(
    '--activation', type=str,default='relu'
)
@click.option(
    '--loss_func', type=str, help = loss_help,default='mse'
)
@click.option(
    '--Nmax', 'Nmax', type=int, help = 'max number of layers',default=4
)
@click.option(
    '--Nmin', 'Nmin', type=int, help = 'min number of layers',default=1
)
def fit_pol(optimizer, activation, loss_func, Nmax, Nmin):


    # create dirs structure 
    os.makedirs(optimizer+"/"+activation, exist_ok=True)
    

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


    # if the file exist restore previous data 
    #and eventually overwrite them

    # combinations
    comb_data=[]
    if os.path.exists(optimizer+"/"+activation+"/"+combfile):
        comb_data = open(
            optimizer+"/"+activation+"/"+combfile,
            'r'
        ).readlines()
    # open file to write combinations of neuron per layer
    else:
        with open(optimizer+"/"+activation+"/"+combfile, "w") as myfile:
            myfile.write("best_combinations\n")
            comb_data.append("best_combinations\n")

    # overwrite prev def
    if os.path.exists(optimizer+"/"+activation+"/"+lossfile):
        bl    = pd.read_csv(optimizer+"/"+activation+"/"+lossfile)
        print(bl)
    if os.path.exists(optimizer+"/"+activation+"/"+predfile):
        bpred = pd.read_csv(optimizer+"/"+activation+"/"+predfile)
    if os.path.exists(optimizer+"/"+activation+"/"+parafile):
        bpar  = pd.read_csv(optimizer+"/"+activation+"/"+parafile)


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
        bl.to_csv(optimizer+"/"+activation+"/"+lossfile, index=False)
        bpred.to_csv(optimizer+"/"+activation+"/"+predfile, index=False)
        bpar.to_csv(optimizer+"/"+activation+"/"+parafile, index=False)
        
        if k>=len(comb_data):
            comb_data.append(" ".join(map( str, comb[bestindex])) + "\n")
        else:
            comb_data[k] = " ".join(map( str, comb[bestindex]))+"\n"
        #with open(optimizer+"/"+activation+"/"+combfile, "a") as myfile:
        #    myfile.write(" ".join(map( str, comb[bestindex])))
        #    myfile.write("\n")

    with open(optimizer+"/"+activation+"/"+combfile, "w") as myfile:
        myfile.writelines(comb_data)


@click.command(
    'compare_sinfit', help='compare NN performance along white noise value in train data'
)
@click.option(
    '-opt', type=str, default='sgd'
)
def compare_sinfit(opt):

    archs = {
        '2x4':{},'4x2':{},'4x4':{},'4x8':{},
        '8x4':{},'8x8':{}, '8x16':{}, 
        '16x8':{}, '16x16':{}
    }

    for key in archs:
        comb = np.array(key.split('x'), dtype=int)
        archs[key] = {
            0.0: {},
            0.1: {},
            0.5: {},
            1.0: {}
        }
        print('training at different sigma the configuration '+key)
        for sigma in tqdm([0.0, 0.1, 0.5, 1.0]):
            mod, hist = fitsin(neurs=comb.tolist(), opt=opt, sigma=sigma)

            archs[key][sigma] = {'mod':mod, 'hist': hist}

            # add saving for model parameters and history

    plotperf(archs)


cli.add_command(fit_pol)

if __name__=='__main__':
    cli()