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

from common import *

OCC_GRID = get_occupancy_grid()

class PathSubscriber(object):
  def __init__(self, name='c0'):
    rospy.Subscriber('/'+name+'/path', PoseArray, self.callback)
    self._path = None

  def callback(self, msg):
    self._path = np.array([[pose.position.x, pose.position.y] for pose in msg.poses])

  @property
  def ready(self):
    return not self._path is None

  @property
  def path(self):
    return self._path

def in_line_of_sight(p1, p2):
  sample_rate = 10 # Samples per meter
  n_samples = np.linalg.norm(p1 - p2) * sample_rate
  x = np.linspace(p1[X], p2[X], n_samples)
  y = np.linspace(p1[Y], p2[Y], n_samples)
  for coord in zip(x, y):
    if OCC_GRID.is_occupied(coord):
      return False
  return True


def run(args):
  c_id = args.id
  c_name = 'c' + str(c_id)
  rospy.init_node('chaser' + str(c_id))

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/' + c_name + '/cmd_vel', Twist, queue_size=5)
  if RVIZ_PUBLISH:
    rviz_publisher = rospy.Publisher('/' + c_name + '_position', PointStamped, queue_size=1)

  runners = ['r0', 'r1', 'r2']

  groundtruth = MultiGroundtruthPose([c_name] + runners)
  path_sub = PathSubscriber(c_name)


  frame_id = 0
  init_publish = False
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready:
      rate_limiter.sleep()
      continue

    pose = groundtruth.poses[c_name]

    # Publish initial position
    if not init_publish or not path_sub.ready:
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

    # Calculate and publish control inputs.
    v = get_velocity(pose[:2], path_sub.path, CHASER_SPEED)

    for runner in runners:
      runner_pos = groundtruth.poses[runner][:2]
      diff = runner_pos - pose[:2]

      if np.linalg.norm(diff) < 1 and in_line_of_sight(runner_pos, pose[:2]):
        v = diff / np.linalg.norm(diff) * 0.1
        print(c_name, 'beelining')
    
    u, w = feedback_linearized(pose, v, 0.1)
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
