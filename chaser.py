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

from common import MultiGroundtruthPose

class PathSubscriber(object):
  def __init__(self, name='c0'):
    rospy.Subscriber('/'+name+'/path', PoseArray, self.callback)
    self._path = []

  def callback(self, msg):
    self._path = [pose.position for pose in msg.poses]

  @property
  def ready(self):
    return not np.isnan(self._path[0])

  @property
  def path(self):
    return self._path


def path_follow(path, position):


def run(args):
  c_id = args.id
  c_name = 'c' + str(c_id)
  rospy.init_node('chaser' + str(c_id))

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/' + c_name + '/cmd_vel', Twist, queue_size=5)

  groundtruth = MultiGroundtruthPose([c_name])
  path = PathSubscriber(c_name)

  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready or not path.ready:
      rate_limiter.sleep()
      continue

    # Calculate and publish control inputs.
    u, w = path_follow(path, groundtruth.poses[])
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
