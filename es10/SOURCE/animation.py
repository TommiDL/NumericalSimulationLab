import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import click 
import geopandas as gpd

@click.group()
def cli():
    pass

def plot_map(ax):
    italy = gpd.read_file('it.json')
    italy.plot(ax=ax, markersize=.5)
    return

@click.command('gif')
@click.option(
    '--coord-file', type=str, help='file of coordinatesin csv format',
    default='coordinates.csv'
)
@click.option(
    '--sol-file', type=str, help='file of best solutions',
    default='best_sol.dat'
)
@click.option(
    '--gif-name', type=str, help='output gif path',
    default='animation.gif'
)
@click.option(
    '--frames', type=int, help='max num of frame to generate gif', default = 1000
)
@click.option('--figtitle', type=str, default='')
def animation(
    coord_file, sol_file, gif_name,figtitle, frames
):
    coord = pd.read_csv(coord_file, names=['x','y'], sep=' ',)
    print(coord)
    bests = pd.read_csv(sol_file, names=[f'city{k}' for k in range(coord.index.size+1)]+['dist'], skiprows=1)
    print(bests)
    Figure , ax = plt.subplots(2,1,figsize=(5, 8), gridspec_kw={'height_ratios': [5, 3]})

    Figure.suptitle(figtitle)
    plot_map(ax[0])

    ax[1].set_xlim(-1, frames)  
    ax[1].set_ylim(0,bests.dist[0]+10)  


    ax[1].grid(True)
    ax[0].grid(True)

    # creating a plot
    lines_plotted = ax[0].plot([], marker='o', c='r')  
    line_plotted = lines_plotted[0]

    label = ax[0].text(0.8, 0.95, "", ha='center', va='center', fontsize=20, color="Black")
    label2 = ax[1].text(0.8, 0.9, "", ha='center', va='center', fontsize=20, color="Black", transform=ax[1].transAxes)

    dists_plotted = ax[1].plot([])
    dist_plotted = dists_plotted[0]

    # function takes frame as an input
    def AnimationFunction(j): 
        
        x=np.array([coord.iloc[int(k)-1].x for k in bests.iloc[j][:-1].array])
        y=np.array([coord.iloc[int(k)-1].y for k in bests.iloc[j][:-1].array])
        line_plotted.set_data((x, y))  
        label.set_text(f'step {j}')
        label2.set_text(f'dist = {bests.dist[j]}')
        

        dist_plotted.set_xdata(bests.index[:j])
        dist_plotted.set_ydata(bests.dist[:j])


    anim_created = FuncAnimation(
        Figure, AnimationFunction,
        #frames=100, interval=25)    
        frames=frames, interval=25)


    video = anim_created.to_html5_video()
    html = display.HTML(video)
    display.display(html) 
    anim_created.save(
        gif_name,
        fps=25
        )

    plt.close()


cli.add_command(animation)

if __name__=='__main__':
    cli()