import numpy as np
import matplotlib.pyplot as plt
from common import *

i = 0


def world_to_image(x, size, res):
    xi = (x + size) / res
    return xi

def plot_field_path(field, size):
    global i
    print('plotting', i)
    res = 0.1

    samples = np.array([[field.sample(np.array([y,x], dtype=np.float32)) for y in np.arange(-size, size, res)] for x in np.arange(-size, size, res)])

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    plt.imshow(samples, cmap='magma', interpolation='nearest', origin='middle')

    j = 0
    for target in field.targets.values():
        path = target[0]
        if path is None or len(path) == 0:
            continue
        xs = np.array(path)[:,0]
        ys = np.array(path)[:,1]

        xs = np.array([world_to_image(x, size, res) for x in xs])
        ys = np.array([world_to_image(y, size, res) for y in ys])

        color = 'green'
        if j == 1:
            color = 'turquoise'
        
        plt.plot(xs, ys, color, lw=2)
        j += 1

    #plt.colorbar(ticks=[samples.min(), samples.max()])
    plt.savefig('chaser_pf' + str(i) + '.png')
    
    plt.show()
    i += 1


def plot_field(field, size):
    global i
    print('plotting', i)
    res = 0.1

    samples = np.array([[field.sample(np.array([y,x], dtype=np.float32)) for y in np.arange(-size, size, res)] for x in np.arange(-size, size, res)])

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    plt.imshow(samples, cmap='magma', interpolation='nearest', origin='middle')

    j = 0
    for target in field.targets.values():
        path = target[0]
        if path is None or len(path) == 0:
            continue
        xs = np.array(path)[:,0]
        ys = np.array(path)[:,1]

        xs = np.array([world_to_image(x, size, res) for x in xs])
        ys = np.array([world_to_image(y, size, res) for y in ys])

        color = 'green'
        if j == 1:
            color = 'turquoise'
        
        plt.plot(xs, ys, color, lw=2)
        j += 1

    #plt.colorbar(ticks=[samples.min(), samples.max()])
    plt.savefig('chaser_pf' + str(i) + '.png')
    
    plt.show()
    i += 1

if __name__ == '__main__':
    pf = PotentialField({'a':[[[0., 0.], [1.,1.],[2.,0.], [5., -2.], [3, 2]], 1, 1]}, is_path=True)
    plot_field(pf, 8)
