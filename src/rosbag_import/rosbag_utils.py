import rosbag


def print_rosbag_info(rosbag_filename):
    bag = rosbag.Bag(rosbag_filename)
    print("Topics found in this rosbag file:")
    for topic, topictuple in bag.get_type_and_topic_info().topics.items():
        print(f"    Name {str(topic).ljust(36)} Message type {str(topictuple.msg_type).ljust(36)}, "
              f"{bag.get_message_count(topic_filters=topic)} messages")
