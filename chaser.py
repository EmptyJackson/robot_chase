#!/usr/bin/env python
import argparse
import rospy
import numpy as np

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist, PoseArray
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

from common import *

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


def run(args):
  c_id = args.id
  c_name = 'c' + str(c_id)
  rospy.init_node('chaser' + str(c_id))

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/' + c_name + '/cmd_vel', Twist, queue_size=5)

  groundtruth = MultiGroundtruthPose([c_name])
  path_sub = PathSubscriber(c_name)

  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready or not path_sub.ready:
      rate_limiter.sleep()
      continue

    # Calculate and publish control inputs.
    pose = groundtruth.poses[c_name]
    v = get_velocity(pose[:2], path_sub.path)
    u, w = feedback_linearized(pose, v, 0.1)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)
    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs single chaser')
  parser.add_argument('--id', action='store', default=0, type=int, help='Chaser id.', choices=[0, 1, 2])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
