import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd


coord = pd.read_csv('coordinates.csv', skiprows=1)
coord
bests = pd.read_csv('best_sol.dat', skiprows=1, names=[f'city{k}' for k in range(coord.index.size+1)]+['dist'])
#print(bests)

Figure , ax = plt.subplots(2,1,figsize=(5, 8), gridspec_kw={'height_ratios': [5, 3]})



circle1 = plt.Circle((0, 0), 1, color='r', fill=False)

ax[0].add_patch(circle1)
# creating a plot
lines_plotted = ax[0].plot([], marker='o')  

line_plotted = lines_plotted[0]

ax[0].set_xlim(-1.2, 1.2)  
ax[0].set_ylim(-1.2, 1.2)  
ax[1].set_ylim(bests.dist[0]-100,bests.dist[0]+100)  

ax[1].set_xlim(-1, bests.index.size)  

label = ax[0].text(0.8, 0.95, "", ha='center', va='center', fontsize=20, color="Black")
label2 = ax[0].text(0.8, 0.7, "", ha='center', va='center', fontsize=20, color="Black")

dists_plotted = ax[1].plot([])
dist_plotted = dists_plotted[0]

steps=[]
dists=[]
# function takes frame as an input
def AnimationFunction(j): 
    
    x=np.array([coord.iloc[int(k)-1].x for k in bests.iloc[j][:-1].array])
    y=np.array([coord.iloc[int(k)-1].y for k in bests.iloc[j][:-1].array])
    line_plotted.set_data((x, y))  
    label.set_text(f'step {j}')
    label2.set_text(f'dist = {bests.dist[j]}')
    
    steps.append(j)
    dists.append(bests.dist[j])
    dist_plotted.set_data((steps, dists)) 

anim_created = FuncAnimation(
    Figure, AnimationFunction,
    #frames=100, interval=25)    
    frames=bests.index.size, interval=25)


video = anim_created.to_html5_video()
html = display.HTML(video)
display.display(html) 
anim_created.save(
    'animation.gif',
    fps=25
    )

# good practice to close the plt object.
plt.close()
