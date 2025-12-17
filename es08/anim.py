import functools
from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('es/OUTPUT/config2/optimization.dat',sep=", " )
df.head()


data = df.loc[df.TEMP==4]

def animate(frame_number):
    # Simulate new data coming in.
    ax.clear()
    data = df.loc[df.TEMP==df.TEMP.unique()[frame_number]]
    ax.set_title(f'Temp = {df.TEMP.unique()[frame_number]}')
    ax.hist(data.EN, density=True, bins=50)
    fig.canvas.draw()

fig, ax = plt.subplots()
ax.hist(data.EN, density=True, bins=50)

ani = FuncAnimation(fig, animate, frames=len(df.TEMP.unique()))

plt.show()