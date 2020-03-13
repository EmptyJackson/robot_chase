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
    
    best_dir = None
    best_dist = 3.    # Minimum distance of 3 meters for a chaser to target a runner
    for r in RUNNERS:
      r_dir = groundtruth.poses[r][:2] - pose[:2]
      dist = np.linalg.norm(r_dir)
      if dist < best_dist:
        best_dist = dist
        best_dir = r_dir
    
    if not best_dir is None:
      v = best_dir / best_dist
      u, w = feedback_linearized(pose, v, 0.1)
    else:
      u, w = braitenberg(*laser.measurements)

    u /= 10

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
