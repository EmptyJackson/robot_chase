from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
import math

from common import *

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)

MIN_ITERATIONS = 300
MAX_ITERATIONS = 2000
OPEN_ITERATIONS = 700


def sample_random_position(occupancy_grid):
  position = np.zeros(2, dtype=np.float32)

  width = len(occupancy_grid.values) * occupancy_grid.resolution
  height = len(occupancy_grid.values[0]) * occupancy_grid.resolution
  o_x = occupancy_grid.origin[0]
  o_y = occupancy_grid.origin[1]
  
  position[0] = np.random.random() * width + o_x
  position[1] = np.random.random() * height + o_y

  while not occupancy_grid.is_free(position):
    position[0] = np.random.random() * width + o_x
    position[1] = np.random.random() * height + o_y
  
  return position


def equal_sign(s1, s2):
  return int(s1[0]) == int(s2[0]) and int(s1[1]) == int(s2[1])

def next_sign(sign):
  return np.array([sign[1], -sign[0]])

def prev_sign(sign):
  return next_sign(next_sign(next_sign(sign)))

def adjust_pose(node, final_position, occupancy_grid):
  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_node = Node(final_pose)

  beta, arc_length = find_angle_with_obstacles(node.pose[:2], final_position, node.pose[2], occupancy_grid)

  if beta is None:
    return None

  final_node.pose[2] = beta
  final_node.cost = node.cost + arc_length
  final_node.local_cost = arc_length
  return final_node


def find_midway_nodes(A, B, theta, occupancy_grid):
  _, angle_add = find_angle_with_obstacles(A, B, theta, occupancy_grid)

  beta, C, r, clockwise = find_angle(A, B, theta)

  C_to_node = (A - C)
  C_to_final = (B - C)

  starting_angle = math.atan2(C_to_node[1], C_to_node[0])
  
  end_angle = math.atan2(C_to_final[1], C_to_final[0])

  direction = 1
  if clockwise:
    direction = -1

  mid_nodes = []

  for ap in [0.25, 0.5, 0.75]:
    angle = (starting_angle + (angle_add * ap * direction) + np.pi * 2) % (np.pi * 2)
    x = C[0] + r * np.cos(angle)
    y = C[1] + r * np.sin(angle)

    mid_nodes.append(np.array([x, y], dtype=np.float32))

  return mid_nodes

def find_angle_with_obstacles(A, B, theta, occupancy_grid):
  beta, C, r, clockwise = find_angle(A, B, theta)

  C_to_node = (A - C)
  C_to_final = (B - C)

  starting_angle = math.atan2(C_to_node[1], C_to_node[0])
  
  end_angle = math.atan2(C_to_final[1], C_to_final[0])

  direction = 1
  if clockwise:
    direction = -1

  da = (occupancy_grid.resolution / r) * direction * 2.5
  angle_add = 0
  while abs((starting_angle + angle_add + 2 * np.pi * 2 - end_angle) % (np.pi * 2)) > abs(da * 2):
    angle = (starting_angle + angle_add + np.pi * 2) % (np.pi * 2)

    x = C[0] + r * np.cos(angle)
    y = C[1] + r * np.sin(angle)

    if not occupancy_grid.is_free(np.array([x, y])):
      return None, 0

    angle_add += da


  return beta, abs(angle_add)

def find_angle(A, B, theta):
    phi = theta + np.pi / 2.

    # P - line going through C and A
    P_m = np.tan(phi)
    P_c = A[1] - P_m * A[0]

    AB_mid = (A + B) / 2.
    A_to_B = B - A

    alpha_dir = np.array([-A_to_B[1], A_to_B[0]], dtype=np.float32)

    # Q - line going through C and midpoint(A,B)
    Q_m = alpha_dir[1] / alpha_dir[0]
    Q_c = AB_mid[1] - Q_m * AB_mid[0]

    # C - centre of circle
    C = np.array([None, None], dtype=np.float32)
    C[0] = (Q_c - P_c) / (P_m - Q_m)
    C[1] = Q_m * C[0] + Q_c

    r = np.linalg.norm(A - C)

    C_to_B = B - C
    beta_dir = np.array([-C_to_B[1], C_to_B[0]], dtype=np.float32)

    beta = np.arctan(beta_dir[1] / beta_dir[0])

    A_sign = np.sign(A - C)
    B_sign = np.sign(B - C)
    theta_sign = np.sign([np.cos(theta), np.sin(theta)])
    beta_sign = np.sign([np.cos(beta), np.sin(beta)])

    A_clockwise = equal_sign(theta_sign, next_sign(A_sign))
    B_clockwise = equal_sign(beta_sign, next_sign(B_sign))

    if A_clockwise != B_clockwise:
      beta += np.pi

    return beta, C, r, A_clockwise


class SampleGrid(object):

  def __init__(self, world_size, world_origin, tile_size):
    self.world_size = world_size
    self.world_origin = world_origin
    self.tile_size = tile_size

    self.grid_size = np.ceil(world_size / tile_size).astype(int)

    self.grid = [[[] for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]

    self.possible_sampling_tiles = []

  def world_to_grid(self, position):
    pos = position - self.world_origin
    indexes = np.floor(pos / self.tile_size)
    index = np.array([int(indexes[0]), int(indexes[1])]).astype(int)
    return index

  def grid_to_world(self, index):
    return (np.array(index, dtype=np.float32) * self.tile_size) + self.world_origin
    

  def add_node(self, node):
    index = self.world_to_grid(node.position)
    try:
      self.grid[index[0]][index[1]].append(node)
    except:
      print('!!!', index, node.position)
      raise
    
    if len(self.grid[index[0]][index[1]]) == 1:
      for lx in [-1, 0, 1]:
        for ly in [-1, 0, 1]:
          x = index[0] + lx
          y = index[1] + ly

          if self.in_bounds(x, y):
            if not [x, y] in self.possible_sampling_tiles:
              self.possible_sampling_tiles.append([x, y])
    

            
  def in_bounds(self, x, y):
    return x >= 0 and x < self.grid_size[0] and y >= 0 and y < self.grid_size[1]

  def is_point_in_possible_tile(self, position):
    index = self.world_to_grid(position)
    for t in self.possible_sampling_tiles:
      if (t == index).all():
        return True
    return False

  def get_close_nodes(self, position):
    nodes = []
    index = self.world_to_grid(position)
    
    for lx in [-1, 0, 1]:
      for ly in [-1, 0, 1]:
        x = index[0] + lx
        y = index[1] + ly

        if self.in_bounds(x, y):
          for node in self.grid[x][y]:
            nodes.append(node)

    return nodes

  def sample_line_gauss(self, p1, p2, sigma):
    r = np.random.random()
    line_point = r * p1 + (1 - r) * p2

    diff = (p2-p1) / np.linalg.norm(p2 - p1)
    perp = np.array([diff[1], -diff[0]], dtype=np.float32)
    
    p = perp * np.random.normal(0, sigma * np.linalg.norm(p2 - p1))
    return line_point + p

  def get_tiles_to_sample(self):
    h = -1
    minTVal = 999999
    minT = None
    for tile in self.possible_sampling_tiles:
      ntiles = len(self.grid[tile[0]][tile[1]])
      if ntiles > h:
        h = ntiles

      if ntiles < minTVal:
        minTVal = ntiles
        minT = tile

    h = h // 4

    ts = [minT]
    for tile in self.possible_sampling_tiles:
      ntiles = len(self.grid[tile[0]][tile[1]])
      if ntiles < h:
        ts.append(tile)

    return ts

  def rpoint(self, start=None, goal=None, sigma=None):
    if start is None or np.random.random() < 0.2:
      sampling_tiles = self.possible_sampling_tiles
      
      ti = np.random.randint(len(sampling_tiles))
      tile = sampling_tiles[ti]

      base_position = self.grid_to_world(tile)

      position = np.array(base_position, dtype=np.float32) + (np.random.random(2) * self.tile_size)
      return position

    else:
      position = self.sample_line_gauss(start, goal, sigma)
      while not self.is_point_in_possible_tile(position):
        position = self.sample_line_gauss(start, goal, sigma)

      return position
      

  def sample_random_point(self, occupancy_grid, start=None, goal=None, sigma=None):
    position = self.rpoint(start, goal, sigma)
    while not occupancy_grid.is_free(position):
      position = self.rpoint(start, goal, sigma)

    return position

  def rclose_point(self, p):
    ra = [(np.random.random() * 2. - 1.) * 2 for _ in range(2)]
    return p + ra
  
  def sample_random_close_point(self, p, occupancy_grid):
    position = self.rclose_point(p)
    while not occupancy_grid.is_free(position):
      position = self.rclose_point(p)

    return position

    
# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  nIndex = 0
  
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.
    self.local_cost = 0
    self.killed = False

    self.pf_sum = 0
    self.pf_local = 0

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c


  def update_parent(self, new_parent, new_local_cost, occupancy_grid, potential_field, targets=None):
    if self.neighbors:
      return
    if self.killed:
      return
    
    if self.parent:
      self.parent.neighbors.remove(self)
    self.parent = new_parent
    self.parent.neighbors.append(self)

    self.pf_local = potential_field.sample(self.pose[:2], targets)
    self.pf_local *= self.pf_local
    self.pf_sum = self.parent.pf_sum + self.pf_local

    self.cost = self.parent.cost + new_local_cost + self.pf_local
    self.local_cost = new_local_cost

    try:
      self._pose = adjust_pose(self.parent, self.pose[:2], occupancy_grid).pose
    except:
      self.kill()
      return

    for n in self.neighbors:
      n.update_prev_cost(self.cost, occupancy_grid)

  def update_prev_cost(self, new_prev_cost, occupancy_grid):
    # Not being used
    print('error! - update_prev_cost shouldnt be used')
    return
    
    if self.killed:
      return
    
    self.cost = new_prev_cost + self.local_cost

    try:
      self._pose = adjust_pose(self.parent, self.pose[:2], occupancy_grid).pose
    except:
      self.kill()
      return
    
    for n in self.neighbors:
      n.update_prev_cost(self.cost, occupancy_grid)


  def kill(self):
    if self.parent:
      self.parent.neighbors.remove(self)
    self.killed = True
    for n in self.neighbors:
      n.kill()

  def steps(self):
    if not self.parent:
      return 1
    return 1 + self.parent.steps()


def rrt_star(start_pose, goal_position, occupancy_grid, potential_field, is_open=False, targets=None):  
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  start_node.pf_sum = potential_field.sample(start_pose[:2]) ** 2
  final_node = None

  world_width = len(occupancy_grid.values) * occupancy_grid.resolution
  world_height = len(occupancy_grid.values[0]) * occupancy_grid.resolution
  world_size = np.array([world_width, world_height], dtype=np.float32)

  MAX_DISTANCE_BETWEEN_NODES = 1.5

  sample_grid = SampleGrid(world_size, occupancy_grid.origin, MAX_DISTANCE_BETWEEN_NODES + 0.5)

  found = False
  
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  
  graph.append(start_node)
  sample_grid.add_node(start_node)

  for i in range(OPEN_ITERATIONS if is_open else MAX_ITERATIONS):

    if i < 50:
      position = sample_grid.sample_random_close_point(start_node.position, occupancy_grid)
    else:
      if not is_open:
        position = sample_grid.sample_random_point(occupancy_grid, start_node.position, goal_position, 0.2)
      else:
        position = sample_grid.sample_random_point(occupancy_grid)
    #too_close = False
    #for node in sample_grid.get_close_nodes(position):
    #  if np.linalg.norm(node.position - position) < .2:
    #    too_close = True
    #    break
    #if too_close:
    #  continue

    # With a random chance, draw the goal position.
    if (np.random.rand() < .05 or i == 0) and not is_open:
      position = goal_position
    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in sample_grid.get_close_nodes(position)), key=lambda x: x[1])
    # Pick a node at least some distance away but not too far.
    # We also verify that the angles are aligned (within pi / 4).
    u = None
    lowest_cost = 999999
    for n, d in potential_parent:
      if d < .2:
        break
      if d > .2 and d < MAX_DISTANCE_BETWEEN_NODES and n.direction.dot(position - n.position) / d > 0.70710678118:
        try:
          beta, arc_length = find_angle_with_obstacles(n.pose[:2], position, n.pose[2], occupancy_grid)
        except Exception as e:
          print('AAAAAA', e)
          continue

        if not beta is None:
          cost = arc_length + n.cost
          if not is_open:
            cost += n.pf_sum
          if cost < lowest_cost:
            lowest_cost = cost
            u = n

    if u is None:
      continue
    
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
        
    
    
    u.add_neighbor(v)
    v.parent = u
    
    for node in sample_grid.get_close_nodes(v.pose[:2]):
      try:
        beta, arc_length = find_angle_with_obstacles(v.pose[:2], node.pose[:2], v.pose[2], occupancy_grid)
      except Exception as e:
        print('AAAAAA', e)
        continue
      if not beta is None:
        if not is_open:
          pfparent = node.parent.pf_sum if (not node.parent is None) else 0
          if arc_length + v.pf_sum < node.local_cost + pfparent:
            node.update_parent(v, arc_length, occupancy_grid, potential_field, targets)
        elif is_open and arc_length < node.local_cost:
          node.update_parent(v, arc_length, occupancy_grid, potential_field, targets)

    v.pf_local = potential_field.sample(v.position, targets) ** 2
    v.pf_sum = v.pf_local + u.pf_sum

    for node in graph[:]:
      if node.killed:
        graph.remove(node)
    
    graph.append(v)
    sample_grid.add_node(v)
    
    if np.linalg.norm(v.position - goal_position) < .2 and not is_open:
      final_node = v
      found = True
    
    if i > MIN_ITERATIONS and found:
      break

  if is_open:
    ld_node = None
    ld_val = np.inf

    for node in graph:
      if node.pf_sum < ld_val:
        ld_val = node.pf_sum / node.steps()
        ld_node = node
    print(ld_node.position)
        
    return start_node, ld_node
  else:
    return start_node, final_node


def rrt_star_path(start_pose, goal_position, occupancy_grid, potential_field, is_open=False, targets=None):
  start_node, final_node = rrt_star(start_pose, goal_position, occupancy_grid, potential_field, is_open, targets)

  if final_node is None:
    return [], start_node, final_node

  path_nodes_rev = [final_node]

  current_node = final_node
  while current_node.parent is not None:
    path_nodes_rev.append(current_node.parent)
    current_node = current_node.parent

  path = list(reversed([node.pose for node in path_nodes_rev]))

  path_extended = []
  for i, node in enumerate(path):
    path_extended.append(path[i][:2])

    if i != len(path)-1:
      next_node = path[i+1]

      mid_nodes = find_midway_nodes(node[:2], next_node[:2], node[2], occupancy_grid)
      for mid_pos in mid_nodes:
        path_extended.append(mid_pos)

  ds = np.array([], dtype=np.float32)
  for i, n in enumerate(path_extended):

    if i != len(path_extended)-1:
      next_node = path_extended[i+1]

      ds = np.append(ds, np.linalg.norm(next_node - n))


  #print('mean,min,max distance between path points extended', ds.mean(), ds.min(), ds.max())

  path_stripped = [path_extended[0]]
  last_node = path_extended[0]

  for i, n in enumerate(path_extended):
    if i != 0 and i != len(path_extended)-1:
      dist = np.linalg.norm(n - last_node)
      if dist > 0.3:
        path_stripped.append(n)
        last_node = n

  path_stripped.append(path_extended[-1])


  ds = np.array([], dtype=np.float32)
  for i, n in enumerate(path_stripped):

    if i != len(path_stripped)-1:
      next_node = path_stripped[i+1]

      ds = np.append(ds, np.linalg.norm(next_node - n))

  #print('mean,min,max distance between path points stripped', ds.mean(), ds.min(), ds.max())

  return path_stripped, start_node, final_node
  

def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  # Run RRT.
  start_node, final_node = rrt_star(START_POSE, GOAL_POSITION, occupancy_grid)

  # Plot environment.
  fig, ax = plt.subplots()
  occupancy_grid.draw()
  plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  draw_solution(start_node, final_node)
  plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)
  
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - 2., 2. + .5])
  plt.ylim([-.5 - 2., 2. + .5])
  plt.show()
  
