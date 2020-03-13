#!/usr/bin/env python
import argparse
import rospy
import numpy as np

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist, PoseArray, PointStamped, Point
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from std_msgs.msg import String

from common import *

OCC_GRID = get_occupancy_grid()
RUNNERS = ['r0', 'r1', 'r2']

class SimpleLaser(object):
  def __init__(self):
    rospy.Subscriber('/c0/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b;

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)

    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)

  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements


def braitenberg(front, front_left, front_right, left, right):
  s = np.tanh([left, front_left, front, front_right, right])
  s = [1 - x for x in s]

  u_weights = [0, -0.5, -1, -0.5, 0]
  w_weights = [-0.5, -1, 0, 1, 0.5]

  u = np.dot(s, u_weights) + 1.
  w = np.dot(s, w_weights)

  return u, w

def rule_based(front, front_left, front_right, left, right):
  u = 0.
  w = 0.

  SLOW_SPEED = 0.1
  FAST_SPEED = 0.3
  TURN_SPEED = 0.6

  d = 0.5
  vcd = 0.15

  f = front < d
  fl = front_left < d or left < d / 3.
  fr = front_right < d or right < d / 3.

  very_close = front < vcd or front_left < vcd or front_right < vcd

  #111
  if f and fl and fr:
    if very_close:
      u = -SLOW_SPEED
      w = 0
    else:
      u = SLOW_SPEED
      w = TURN_SPEED
  #110
  elif f and fl and not fr:
    u = SLOW_SPEED
    w = -TURN_SPEED
  #011
  elif f and not fl and fr:
    u = SLOW_SPEED
    w = TURN_SPEED
  #010
  elif f and not fl and not fr:
    if very_close:
      u = -SLOW_SPEED
      w = 0
    else:
      u = SLOW_SPEED
      w = TURN_SPEED
  #101
  elif not f and fl and fr:
    if very_close:
      u = -SLOW_SPEED
      w = 0
    else:
      u = SLOW_SPEED
      w = TURN_SPEED
  #100
  elif not f and fl and not fr:
    u = SLOW_SPEED
    w = -TURN_SPEED
  #001
  elif not f and not fl and fr:
    u = SLOW_SPEED
    w = TURN_SPEED
  #000
  elif not f and not fl and not fr:
    u = FAST_SPEED
    w = 0
  else:
    print('err invalid sensor config:', f, fl, fr)
    
  return u, w

def capture_runner(r_name_msg):
  RUNNERS.remove(r_name_msg.data)


def run(args):
  c_id = args.id
  c_name = 'c' + str(c_id)
  rospy.init_node('chaser' + str(c_id))

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/' + c_name + '/cmd_vel', Twist, queue_size=5)
  if RVIZ_PUBLISH:
    rviz_publisher = rospy.Publisher('/' + c_name + '_position', PointStamped, queue_size=1)

  rospy.Subscriber('/captured_runners', String, capture_runner)
  
  groundtruth = MultiGroundtruthPose([c_name] + RUNNERS)
  laser = SimpleLaser()

  frame_id = 0
  init_publish = False

  print('Chaser controller initialized.')
  
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not laser.ready or not groundtruth.ready:
      rate_limiter.sleep()
      continue
    pose = groundtruth.poses[c_name]

    # Publish initial position
    if not init_publish:
      if RVIZ_PUBLISH:
        position_msg = PointStamped()
        position_msg.header.seq = frame_id
        position_msg.header.stamp = rospy.Time.now()
        position_msg.header.frame_id = '/odom'
        pt = Point()
        pt.x = pose[X]
        pt.y = pose[Y]
        pt.z = .05
        position_msg.point = pt
        rviz_publisher.publish(position_msg)
      init_publish = True
      rate_limiter.sleep()
      continue
    
    u, w = braitenberg(*laser.measurements)
    
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    if RVIZ_PUBLISH:
      position_msg = PointStamped()
      position_msg.header.seq = frame_id
      position_msg.header.stamp = rospy.Time.now()
      position_msg.header.frame_id = '/odom'
      pt = Point()
      pt.x = pose[X]
      pt.y = pose[Y]
      pt.z = .05
      position_msg.point = pt
      rviz_publisher.publish(position_msg)

    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs single chaser')
  parser.add_argument('--id', action='store', default=0, type=int, help='Chaser id.', choices=[0, 1, 2])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
