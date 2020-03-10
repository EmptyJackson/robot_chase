import numpy as np
import matplotlib.pyplot as plt

def plot_field(field, size):
    print('plotting')
    res = 0.5

    samples = np.array([[field.sample(np.array([x,y], dtype=np.float32)) for y in np.arange(-size, size, res)] for x in np.arange(-size, size, res)])
    
    plt.imshow(samples, cmap='magma', interpolation='nearest', origin='middle')
    plt.colorbar(ticks=[samples.min(), samples.max()])
    plt.show()
