import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import click 



@click.group()
def cli():
    pass


def plot_circle(ax):
    circle1 = plt.Circle((0, 0), 1, color='r', fill=False)
    ax[0].add_patch(circle1)
    ax[0].set_xlim(-1.2, 1.2)  
    ax[0].set_ylim(-1.2, 1.2)  
    return

def plot_square(ax):
    square1 = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
    ax[0].add_patch(square1)
    ax[0].set_xlim(-0.2, 1.2)  
    ax[0].set_ylim(-0.2, 1.2)  
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
    '--shape', type=str, help = 'type of shape',
    default='circle'
)
@click.option('--figtitle', type=str, default='')
def animation(
    coord_file, sol_file, gif_name, shape,figtitle
):
    coord = pd.read_csv(coord_file, skiprows=1)
    coord
    bests = pd.read_csv(sol_file, skiprows=1, names=[f'city{k}' for k in range(coord.index.size+1)]+['dist'])

    Figure , ax = plt.subplots(2,1,figsize=(5, 8), gridspec_kw={'height_ratios': [5, 3]})

    Figure.suptitle(figtitle)

    if(shape == 'circle'):
        plot_circle(ax)
    else:
        plot_square(ax)


    ax[1].set_xlim(-1, bests.index.size)  
    ax[1].set_ylim(0,bests.dist[0]+10)  


    ax[1].grid(True)
    ax[0].grid(True)

    # creating a plot
    lines_plotted = ax[0].plot([], marker='o')  
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
        frames=bests.index.size, interval=25)


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