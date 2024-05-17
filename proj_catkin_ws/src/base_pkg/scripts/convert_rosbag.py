import rosbag
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

# Define a function to convert Odometry messages to PoseStamped messages
def convert_odometry_to_pose_stamped(odom_msg):
    pose_stamped_msg = PoseStamped()
    pose_stamped_msg.header = odom_msg.header
    pose_stamped_msg.pose.position = odom_msg.pose.pose.position
    pose_stamped_msg.pose.orientation = odom_msg.pose.pose.orientation
    return pose_stamped_msg

# Open the original rosbag file and the new rosbag file
with rosbag.Bag('odom.bag', 'r') as original_bag, rosbag.Bag('odom_converted.bag', 'w') as converted_bag:
    for topic, msg, t in original_bag.read_messages():
        # Check if the message is of type Odometry and the topic is /pose
        if topic == '/pose' and isinstance(msg, Odometry):
            # Convert the Odometry message to PoseStamped
            pose_stamped_msg = convert_odometry_to_pose_stamped(msg)
            # Write the converted message to the new rosbag file
            converted_bag.write(topic, pose_stamped_msg, t)
        else:
            # Write other messages unchanged to the new rosbag file
            converted_bag.write(topic, msg, t)
