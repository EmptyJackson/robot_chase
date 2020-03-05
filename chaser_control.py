#!/usr/bin/env python
import rospy
from common import MultiGroundtruthPose

from geometry_msgs.msg import PoseArray, Pose, Point

def simple(poses):
  # Return paths as array of Point
  pass

def create_pose_array(path):
  pose_path = []
  for position in path:
    p = Pose()
    p.position = position
    pose_path.append(p)
  path_msg = PoseArray()
  path_msg.poses = pose_path
  return path_msg

def run(args):
  rospy.init_node('chaser_control')
  nav_method = globals()[args.mode]

  # Update paths every 100 ms.
  rate_limiter = rospy.Rate(10)
  publishers = 
    [rospy.Publisher('/c'+str(i)+'/path', PoseArray, queue_size=1)]

  gts = MultiGroundtruthPose(['c0', 'c1', 'c2', 'r0', 'r1', 'r2'])

  while not rospy.is_shutdown():
    # Make sure all groundtruths are ready.
    if not gts.ready:
      rate_limiter.sleep()
      continue

    print(gts.poses)

    # todo: Calculate paths
    paths = nav_method(gts.poses)

    for path, publisher in zip(paths, publishers):
      path_msg = create_pose_array(path)
      publisher.publish(path_msg)

    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs centralised chaser control')
  parser.add_argument('--mode', action='store', default='simple', help='Routing method.', choices=['simple'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
