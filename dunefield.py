
import numpy
from matplotlib import pyplot
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm
import time

from  Dunefield_CPU.Debug.cpu_dunefield import CPU_Field 
from  Dunefield_GPU.Debug.gpu_dunefield import GPU_Field 

field_type = {
    'GPU': GPU_Field,
    'CPU': CPU_Field
}

def simulate(width, length, n, f_type):
    field = field_type[f_type]()
    field.initialize(width, length)
    
    time = []

    for _ in tqdm(range(n)):
        # field.simulate_frame()
        time.append(field.simulate_frame())
        
    print("Average time taken per frame: %f" % numpy.mean(time))
    print("Overall time taken: %f" % sum(time))

    print("Number of blocks changed: %i" % field.check_block_level())

    pyplot.imshow(numpy.array(field.get_heights()).reshape([width, length]), cmap='hot', interpolation='nearest')
    pyplot.show()

def animate(width, length, f_type):
    field = field_type[f_type]()
    field.initialize(width, length)

    fig, ax = pyplot.subplots()
    # ax = sns.heatmap(numpy.array(field.get_heights()).reshape([width,length]), square=True, cbar=True)
    ax.imshow(numpy.array(field.get_heights()).reshape([width,length]), cmap='hot', interpolation='nearest')

    def frames():
        while True:
            # time.sleep(20)
            yield 0

    def func(args):
        for _ in range(10): field.simulate_frame()
        # ax = sns.heatmap(numpy.array(field.get_heights()).reshape([width,length]), square=True, cbar=False)
        ax.imshow(numpy.array(field.get_heights()).reshape([width,length]), cmap='hot', interpolation='nearest')

    anim = animation.FuncAnimation(fig, func, frames=frames, interval=200)
    pyplot.show()

if __name__ == '__main__':
    simulate(width = 250, length = 250, n = 10000, f_type="GPU")
    # animate(100,100,"CPU")