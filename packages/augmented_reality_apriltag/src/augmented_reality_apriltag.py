#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer
from cv_bridge import CvBridge
import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import copy
from PIL import Image
import rospkg 
from dt_apriltags import Detector, Detection

"""
This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.
"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = "csc22945"

        self.rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(self.rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')

        # initialise for image conversion
        self.bridge = CvBridge()

        # find the calibration parameters
        self.camera_info_dict = self.load_intrinsics()
        rospy.loginfo(f'Camera info: {self.camera_info_dict}')
        self.homography = self.load_extrinsics()
        rospy.loginfo(f'Homography: {self.homography}')
        rospy.loginfo("Calibration parameters extracted.")

        # extract parameters from camera_info_dict for apriltag detection
        f_x = self.camera_info_dict['camera_matrix']['data'][0]
        f_y = self.camera_info_dict['camera_matrix']['data'][4]
        c_x = self.camera_info_dict['camera_matrix']['data'][2]
        c_y = self.camera_info_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]
        K_list = self.camera_info_dict['camera_matrix']['data']
        self.K = np.array(K_list).reshape((3, 3))
        print(f'K: {self.K}')

        # initialise the apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

        # construct publisher to images
        image_pub_topic = f'/{self.veh}/{node_name}/augmented_image/image/compressed'
        self.image_pub = rospy.Publisher(image_pub_topic, CompressedImage, queue_size=16)
        rospy.loginfo(f'Publishing to: {image_pub_topic}')

        # construct subscriber to images
        image_sub_topic = f'/{self.veh}/camera_node/image/compressed'
        self.image_sub = rospy.Subscriber(image_sub_topic, CompressedImage, self.callback)
        rospy.loginfo(f'Subscribed to: {image_sub_topic}')


    def callback(self, image_msg):
        """ Once recieving the image, save the data in the correct format
            is it okay to do a lot of computation in the callback?
        """
        rospy.loginfo('Image recieved, running callback.')

        # extract the image message to a cv image
        image_np = self.readImage(image_msg)

        # detect apriltag and extract its reference frame
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(image_gray, estimate_tag_pose=False, camera_params=self.camera_params, tag_size=0.065) # returns list of detection objects

        for tag in tags:
            H = tag.homography # assume only 1 tag in image

            # Find transformation from april tag to target image frame
            P = self.projection_matrix(self.K, H)

            # Project model into image frame
            image_np = self.renderer.render(image_np, P)

        # make new CompressedImage to publish
        augmented_image_msg = CompressedImage()
        augmented_image_msg.header.stamp = rospy.Time.now()
        augmented_image_msg.format = "jpeg"
        augmented_image_msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()

        # Publish new image
        self.image_pub.publish(augmented_image_msg)

        rospy.loginfo('Callback completed, publishing image')


    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        self.frame_id = self.veh + '/camera_optical_frame'
        self.cali_file = cali_file_folder + self.veh + ".yaml"

        self.cali_file = self.rospack.get_path('augmented_reality_apriltag') + f"/config/calibrations/camera_intrinsic/{self.veh}.yaml"

        # Locate calibration yaml file or use the default otherwise
        rospy.loginfo(f'Looking for calibration {self.cali_file}')
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        calib_data = self.readYamlFile(self.cali_file)
        self.log("Using calibration file: %s" % self.cali_file)

        return calib_data


    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + self.veh + ".yaml"

        cali_file = self.rospack.get_path('augmented_reality_apriltag') + f"/config/calibrations/camera_extrinsic/{self.veh}.yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(f"Can't find calibration file: {cali_file}.\n Using default calibration instead.", 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        calib_data = self.readYamlFile(cali_file)

        return calib_data['homography']

    
    def projection_matrix(self, K, H):
        """
            K is the intrinsic camera matrix
            H is the homography matrix
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """

        # find R_1, R_2 and t
        Kinv = np.linalg.inv(K)
        r1r2t = np.matmul(Kinv, H) # r1r2t = [r_1 r_2 t]
        r1r2t = r1r2t / np.linalg.norm(r1r2t[:, 0])
        r_1 = r1r2t[:, 0]
        r_2 = r1r2t[:, 1]
        t = r1r2t[:, 2]

        # Find v_3 vector othogonal to v_1 and v_2
        r_3 = np.cross(r_1, r_2)

        # Reconstruct R vector
        R = np.column_stack((r_1, r_2, r_3))

        # Use SVD to make R into an orthogonal matrix
        _, U, Vt = cv2.SVDecomp(R)
        R = U @ Vt

        # Combine R, t and K to find P
        buff = np.column_stack((R, t)) # buff = [r_1 r_2 r_3 t]
        P = K @ buff

        return P


    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []


    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.Loader)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown('No calibration file found.')
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()