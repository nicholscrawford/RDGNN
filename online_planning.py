import rospy
import ros_numpy
import argparse
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float32MultiArray

######################################################################################
# RDGNN Rewrite. 

# The hope is to gain a better understanding of rdgnn implementation.

# The model should be a class that has methods to train, and methods to test.
# CEM sampling should be external, but could be encapuslated into a seperate file.

# It would be best if the model was scalable to various sizes and  actions, so abstracting the idea 
# of an action, and not requiring an input dimension would be best.
#######################################################################################

#Load saved model

#When data comes in:
    #Change to point cloud
    #Sample from point cloud
    #Add one hot encodings
    #Encode to latent state

    #Sampling:
        # Predict future state given action
        # Predict future relations
        # Update best action

    #Return best action


def main(args: argparse.Namespace):
    rdgnn = torch.load(args.saved_model)
        

    rospy.init_node('online_planning')
    points_topic = '/point_state'
    label_img_topic = '/labeled_image'
    rospy.Subscriber(label_img_topic, Image, self.get_label_image)

    point_cloud_topic = '/tf_cloud'
    rospy.Subscriber(point_cloud_topic, PointCloud2, self.get_tf_point_cloud)

    # rospy.Subscriber(points_topic, PointsState, self.get_point_info)

    self.online_action = rospy.Publisher("/online_action", Float32MultiArray, queue_size=1)
    self.online_action_variance = rospy.Publisher("/online_action_variance", Float32MultiArray, queue_size=1)
    self.center_estimation = rospy.Publisher("/center_estimation", Float32MultiArray, queue_size=1)

    self.read_ros_messgage = False
    self.read_ros_messgage_pc = False
    while not rospy.is_shutdown():
        if self.read_ros_messgage and self.read_ros_messgage_pc:
            # print(len(self.point_msg.depth))
            # print(self.point_msg.depth[0])
            # print(np.frombuffer(self.point_msg.depth[0].data, dtype=np.float32).shape)
            label_image = np.frombuffer(self.label_image_msg.data, dtype=np.int8).reshape(self.label_image_msg.height, self.label_image_msg.width)
            print(label_image.shape)
            print('pc shape',self.spatial_points.shape)
            # print(self.spatial_points[0][0])
            # time.sleep(10)
            break




if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
        description='Train for dynamics model from isaacgym data which records sensor data and actions.')
                              
    parser.add_argument('--saved_model', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
