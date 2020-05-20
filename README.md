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




gr = SquareGrid([0, 0, 20, 20], 1)

stat_1 = gr.add_station(1.5, 2.5)
stat_1 = gr.add_station(5.5, 8.5)
stat_1 = gr.add_station(2.5, 4.5)

gr.expand_borders()
fig, ax = plt.subplots(figsize = (10, 10))
gr.plot_grid(ax)
## Where to get it
