-----------------
# graph:

## Example of usage
from matplotlib import animation

gr = SquareGrid([-20, -20, 20, 20], 0.1)

stat_2 = gr.add_station(3.25, 2.75)
stat_3 = gr.add_station(-1.25, -3.25)

fig, ax=plt.subplots(figsize = (10, 10))
container = []

for i in range(17):
    gr.expand_borders()
    container.append(gr.plot_animation(ax, colors = [['w', 'bisque'], ['blue', 'bisque'], ['gray', 'white']]))
ani = animation.ArtistAnimation(fig, container, interval=200, blit=False)

ani.save('grid.mp4')



## Where to get it
