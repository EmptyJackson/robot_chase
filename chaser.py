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
  path_sub = PathSubscriber(c_name)

  rev_frames = 0
  had_path = False

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

    for runner in ['r0', 'r1', 'r2']:
      if np.linalg.norm(groundtruth.poses[runner][:2] - pose[:2]) < 0.15:
        rev_frames = 50

    path = path_sub.path

    if path is None or len(path) == 0:
      if had_path:
        rev_frames = 50

    else:
      had_path = True
    
    
    # Calculate and publish control inputs.
    v = get_velocity(pose[:2], path, CHASER_SPEED)

    for runner in RUNNERS:
      runner_pose = groundtruth.poses[runner]
      runner_pos = runner_pose[:2]
      
      diff = runner_pos - pose[:2]
      dist = np.linalg.norm(diff)
      
      if dist < 1 and in_line_of_sight(runner_pos, pose[:2]):
        f_pos = runner_pos + np.array([np.cos(runner_pose[2]), np.sin(runner_pose[2])]) * dist
        f_pos_diff = f_pos - pose[:2]
        f_pos_dist = np.linalg.norm(f_pos_diff)
        v = f_pos_diff / f_pos_dist * CHASER_SPEED
    
    u, w = feedback_linearized(pose, v, 0.1)

    if rev_frames > 0:
      u = -CHASER_SPEED
      w = 0
    
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
    if rev_frames > 0:
      rev_frames -= 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs single chaser')
  parser.add_argument('--id', action='store', default=0, type=int, help='Chaser id.', choices=[0, 1, 2])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
