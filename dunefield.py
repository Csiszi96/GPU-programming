
import numpy
from matplotlib import pyplot
from matplotlib import animation
import seaborn as sns

from  Dunefield_CPU.Debug.cpu_dunefield import CPU_Field 
from  Dunefield_GPU.Debug.gpu_dunefield import GPU_Field 


width = 500
length = 500

field = GPU_Field()
field.initialize(width, length)

def simulate(n):
    time = numpy.array([])

    for _ in range(n):
        print(".", end="")
        field.simulate_frame()
        # numpy.append(time, field.simulate_frame())
        
    print("Average time taken per frame: %f" % time.mean())
    print("Overall time taken: %f" % time.sum())

    pyplot.imshow(numpy.array(field.get_heights()).reshape([width, length]), cmap='hot', interpolation='nearest')
    pyplot.show()

def animation():
    fig = pyplot.figure()
    ax = sns.heatmap(numpy.array(field.get_heights()).reshape([width,length]), square=True, cbar=True)

    def frames():
        while True:
            yield 0

    def animate(args):
        for _ in range(10): field.simulate_frame()
        ax = sns.heatmap(numpy.array(field.get_heights()).reshape([width,length]), square=True, cbar=False)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=30)
    pyplot.show()

if __name__ == '__main__':
    simulate(10)