""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""
from phi.field._point_cloud import distribute_points
from phi.torch.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


GRAVITY = math.tensor([0, -9.81])
DT = .2
OBSTACLE = Box(x=(1, 25), y=(30, 33)).rotated(-20)
ACCESSIBLE_CELLS = CenteredGrid(~OBSTACLE, 0, x=64, y=64)
ACCESSIBLE_FACES = field.stagger(ACCESSIBLE_CELLS, math.minimum, extrapolation.ZERO)
_OBSTACLE_POINTS = PointCloud(Cuboid(field.support(1 - ACCESSIBLE_CELLS, 'points'), x=2, y=2), color='#000000', bounds=ACCESSIBLE_CELLS.bounds)

particles = distribute_points(union(Box(x=(15, 30), y=(50, 60)), Box(x=None, y=(-INF, 5))), x=64, y=64) * (0, 0)
scene = vis.overlay(particles, _OBSTACLE_POINTS)  # only for plotting


# @math.jit_compile
def step(particles):
    # --- Grid Operations ---
    velocity = prev_velocity = field.finite_fill(StaggeredGrid(particles, 0, x=64, y=64))
    occupied = CenteredGrid(particles.mask(), velocity.extrapolation.spatial_gradient(), velocity.bounds, velocity.resolution)
    velocity, pressure = fluid.make_incompressible(velocity + GRAVITY * DT, [OBSTACLE], active=occupied)
    # --- Particle Operations ---
    particles = flip.map_velocity_to_particles(particles, velocity, prev_velocity)
    particles = advect.points(particles, velocity * ~OBSTACLE, DT, advect.finite_rk4)
    particles = flip.respect_boundaries(particles, [OBSTACLE])
    return particles, velocity, pressure


for _ in view('scene,velocity', display='scene', play=False, namespace=globals()).range():
    particles, velocity, pressure = step(particles)
    scene = vis.overlay(particles.with_values(1), _OBSTACLE_POINTS)  # velocity.vector['y'],
    # vis.show(scene, pressure)
