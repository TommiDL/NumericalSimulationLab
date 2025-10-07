import numpy as np

# generate training inputs
# generate training inputs
def generate_linear_data(m:float, b:float, sigma:float=0.0, seed:int=0, traindim:int=500):
    """
    Generate linear data with gaussian noise
    Parameter:
        - m     [float] : linear weight
        - b     [float] : bias
        - sigma [float] : gaussian noise STD [default 0]
        - seed  [int]   : numpy random seed [default 0]
    Return:
        - x_train  [array], y_train [array] : training dataset
        - x_valid  [array], y_valid [array] : validation dataset
        - y_target [array]                  : noiseless output for x_valid
    """
    np.random.seed(seed)
    x_train = np.random.uniform(-1, 1, traindim)
    x_valid = np.random.uniform(-1, 1, 50)
    x_valid.sort()
    y_target = m * x_valid + b # ideal (target) linear function

    y_train = np.random.normal(m * x_train + b, sigma) # actual measures from which we want to guess regression parameters
    y_valid = np.random.normal(m * x_valid + b, sigma)

    return x_train, y_train, x_valid, y_valid, y_target    
      



def generate_poly_data(pol, seed=0):
    # generate training inputs
    np.random.seed(seed)
    x_train = np.random.uniform(-1, 1, 500)
    x_valid = np.random.uniform(-1, 1, 50)
    x_valid.sort()
    
    
    y_target =  pol(x_valid)# ideal (target) polynomial
    
    sigma = 0.1 # noise standard deviation, for the moment it is absent
    y_train = np.random.normal(pol(x_train), sigma) # actual measures from which we want to guess regression parameters
    y_valid = np.random.normal(pol(x_valid), sigma)

    return x_train, y_train, x_valid, y_valid, y_target

def pol(x):
  return 4-3*x - 2*x**2 + 3*x**3