from model_def import *

def main():

    select:bool=True
    while select:

        n_sigma = int(input("num of sigma >>"))
    
        npoints = int(input("npoints >>"))


        print(f'total iteration: ', n_sigma*npoints)

        if (int(input("press 0 to confirm >> ")) == 0):
            select = False


    grid=regression_search(
        n_sigma=n_sigma,
        n_points=npoints    
    )

    grid.to_csv('regression_grid.csv')


if __name__=='__main__':
    main()