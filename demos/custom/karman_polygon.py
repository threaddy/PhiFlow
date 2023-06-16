from phi.torch.flow import *

# Square, vertices must be ordered
vertices_triang = [[30,30],[50,30],[50,50],[30,50]]

vertices = vertices_triang
vertices= wrap(vertices, channel(vector='x,y'))


pol = infinite_polygon(vertices=vertices)
pol = Obstacle(pol)


SPEED = vis.control(2.)
velocity = StaggeredGrid((SPEED, 0), ZERO_GRADIENT, x=128, y=128, bounds=Box(x=128, y=64))
BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.5), y=None), velocity.extrapolation, velocity.bounds, velocity.resolution)
pressure = None


@jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, p, dt=1.):
    v = advect.semi_lagrangian(v, v, dt)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    return fluid.make_incompressible(v, [pol], Solve('auto', 1e-5, x0=p))


for _ in view('vorticity,velocity,pressure', namespace=globals()).range():
    velocity, pressure = step(velocity, pressure)
    vorticity = field.curl(velocity)
