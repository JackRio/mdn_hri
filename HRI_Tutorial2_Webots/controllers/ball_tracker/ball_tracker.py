import cv2
import numpy as np
import torch
from model import MDN
from torch.autograd import Variable  # storing data while learning

from controller import Robot, Keyboard

model = MDN(2, 20, 2, 5)
ffn_model = torch.load('model/FFN.model')


class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        """
        Initialization class for the Nao Robot.

        :param ext_camera_flag: Indicator for using external camera or not
        :type ext_camera_flag: bool
        """
        super(MyRobot, self).__init__()
        print('> Starting robot controller')

        # Millisecond to process the data (loop frequency)
        # Use int(self.getBasicTimeStep()) for default
        self.timeStep = 32
        self.state = 0  # Idle starts for selecting different states
        # Sensors init
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStep)
        self.step(self.timeStep)  # Execute one step to get the initial position
        # activate camera
        self.cameraBottom = self.getCamera("CameraBottom")
        self.cameraBottom.enable(2 * self.timeStep)
        # load face detector
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.ext_camera = ext_camera_flag
        self.displayCamExt = self.getDisplay('CameraExt')
        if self.ext_camera:
            self.cameraExt = cv2.VideoCapture(0)

        # Actuators init ###ADDED
        self.head_yaw = self.getMotor('HeadYaw')
        self.head_yaw.setPosition(float('inf'))
        self.head_yaw.setVelocity(0)

        self.head_pitch = self.getMotor('HeadPitch')
        self.head_pitch.setPosition(float('inf'))
        self.head_pitch.setVelocity(0)

        self.LShoulder = self.getMotor('LShoulderPitch')
        self.LShoulder.setPosition(float(1.8))
        self.LShoulder.setVelocity(1)

        # -0.3 < 1.3
        self.LShoulderR = self.getMotor('LShoulderRoll')
        self.LShoulderR.setPosition(float(0.3))
        self.LShoulderR.setVelocity(1)

        self.LElbow = self.getMotor('LElbowRoll')
        self.LElbow.setPosition(float(-1.4))
        self.LElbow.setVelocity(1)

        self.RShoulder = self.getMotor('RShoulderPitch')
        self.RShoulder.setPosition(float(1.8))
        self.RShoulder.setVelocity(1)

        # -1.3 < 0.3
        self.RShoulderR = self.getMotor('RShoulderRoll')
        self.RShoulderR.setPosition(float(-0.3))
        self.RShoulderR.setVelocity(1)

        self.RElbow = self.getMotor('RElbowRoll')
        self.RElbow.setPosition(float(1.4))
        self.RElbow.setVelocity(1)

    # Captures the external camera frames
    # Returns the image downsampled by 2
    def camera_read_external(self):
        """
        Captures the picture through external camera

        :return: returns the captured image
        :rtype: nd.array
        """
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5)  # image downsampled by 2

        return img

    def detect_ball(self):
        """
        Looks for the ball in the environment and if it detects the ball returns the co-ordinates of the ball
        :return: Co-ordinates of the ball or None value
        :rtype: int, int
        """

        img = self.cameraBottom.getImage()
        height, width = self.cameraBottom.getHeight(), self.cameraBottom.getWidth()
        # turn into np array
        image = np.frombuffer(img, np.uint8).reshape(height, width, 4)
        # transform into HSV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # green is defined as tripels between these values
        lower = np.array([40, 50, 50])
        upper = np.array([80, 255, 255])
        # lay a mask over all green values
        mask = cv2.inRange(image, lower, upper)
        # Image preparation
        # Erosion
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(mask, kernel, iterations=1)
        # Dilation
        image = cv2.dilate(image, kernel, iterations=1)
        # find ball contour
        contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour) != 0:
            contour = np.asarray(contour[0])
            m = cv2.moments(contour)
            # if area of detected blob is reasonable:
            if 3000 > m["m00"] > 1:
                # calculate momentum
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                return cx, cy
            else:
                return None, None
        return None, None

    def gather_data(self):
        """
        Module to collect data for training the MDN network
            This method collects the co ordinate of the ball and the pitch and roll values.
            The data will be later used to train the MDN and FFN network
        """

        while True:
            records = []
            self.LElbow.setPosition(0)
            self.head_yaw.setVelocity(5)
            self.head_pitch.setVelocity(5)
            self.head_pitch.setPosition(0)
            self.head_yaw.setPosition(0.4)
            roll_position = 0
            pitch_position = 0
            self.LShoulderR.setVelocity(5)
            self.LShoulder.setVelocity(5)
            i = 0
            while True:
                k = self.keyboard.getKey()
                if k == Keyboard.RIGHT:
                    if -0.31 < roll_position + 0.01 < 1.3:
                        roll_position += 0.01
                        self.LShoulderR.setPosition(roll_position)
                if k == Keyboard.LEFT:
                    if -0.31 < roll_position - 0.01 < 1.3:
                        roll_position -= 0.01
                        self.LShoulderR.setPosition(roll_position)
                if k == Keyboard.DOWN:
                    if -2 < pitch_position + 0.01 < 2:
                        pitch_position += 0.01
                        self.LShoulder.setPosition(pitch_position)
                if k == Keyboard.UP:
                    if -2 < pitch_position - 0.01 < 2:
                        pitch_position -= 0.01
                        self.LShoulder.setPosition(pitch_position)
                if k == ord('M'):
                    self.step(400)
                    i += 1
                    print(i)
                    print(pitch_position, roll_position)
                    x, y = self.detect_ball()
                    if x != None:
                        records.append([[pitch_position, roll_position], [x, y]])
                    print(self.detect_ball())
                if k == ord('P'):
                    self.step(400)
                    print(records)
                    records = np.asarray(records)
                    print(records)
                    np.save("data/data.npy", records)
                if self.step(self.timeStep) == -1:
                    break
            return

    @staticmethod
    def sample_pred(pi, sigma, mu):
        """
            Predicts the pitch and roll position for the given co-ordinates take pi, mu and sigma as input
        :param pi: Normalized vector for N mixing coefficients
        :type pi: nd.array
        :param sigma: Variance as a function of x
        :type sigma: nd.array
        :param mu: Mean as a function of x
        :type mu: nd.array
        :return: Prediction array
        :rtype:nd.array
        """
        N, K = pi.shape  # Number of samples and number of Gaussians (Same Pi is assumend to work for all Noutputs)
        _, KT = mu.shape  # Noutput x Number of Gaussians
        NO = int(KT / K)  # Noutput
        pred = Variable(torch.zeros(N, NO))  # sample for all variables

        for i in range(N):  # loop over samples
            r = np.random.uniform(0, 1)  # sample random number from [0, 1) uniform distribution
            summed_pi = 0
            for idx, pi_value in enumerate(pi[i]):
                summed_pi += pi_value.item()
                if summed_pi > r:
                    segment = idx

            for t in range(NO):
                val = np.random.normal(mu[i, t + segment * NO], sigma[i, segment].item())
                pred[i, t] = val
        return pred

    def predict_movement(self, x, y, model_key):
        """
            Main module which takes ball co-ordinates and outputs the roll and pitch location for the robot
            arm. The method uses either FFN or MDN model to predict the values based on the model_key value.
        :param x: Input array of co-ordinates
        :type x: nd.array
        :param y: Output array of position of arm
        :type y: nd.array
        :param model_key: True for MDN and False for FFN
        :type model_key: bool
        :return: Position of roll and pitch
        :rtype: float, float
        """
        # load model 
        model = torch.load('model/MDN.model')
        model.eval()

        # ... normalize input
        x_mean = np.load("data/input_mean.npy")
        x_std = np.load("data/input_std.npy")
        target_mean = np.load("data/target_mean.npy")
        target_std = np.load("data/target_std.npy")

        input_data = np.empty((1, 2))

        input_data[0, 0] = (x - x_mean[0]) / x_std[0]
        input_data[0, 1] = (y - x_mean[1]) / x_std[1]

        input_data = torch.from_numpy((np.float32(input_data)))
        # if using MDN
        if model_key:
            # calculate arm movement
            pi_variable, sigma_variable, mu_variable = model.forward(input_data)
            # print(pi_variable, sigma_variable, mu_variable )

            y_pred = self.sample_pred(pi_variable.data.numpy(), sigma_variable.data.numpy(), mu_variable.data.numpy())
            # de-normalize 
            prediction = (y_pred * target_std) + target_mean
            prediction = prediction.data.numpy()

            return prediction[0, 0], prediction[0, 1]

        # if using FFN
        else:
            prediction = ffn_model(input_data)
            prediction = prediction.data.numpy()
            return prediction[0, 0], prediction[0, 1]

    def follow_ball(self, model_key=True):
        """
            Uses the prediction of the network to move the arm of the Robot to follow the movement of the bot.
        :param model_key: True for MDN and False for FFN (Default is True)
        :type model_key: bool
        """

        # instantiate starting position
        self.head_yaw.setVelocity(1)
        self.head_pitch.setPosition(0)
        self.head_yaw.setPosition(0.4)
        self.LShoulderR.setVelocity(0.3)
        self.LShoulder.setVelocity(0.3)
        self.LElbow.setPosition(0)

        x_ball_old, y_ball_old = self.detect_ball()
        while self.step(self.timeStep) != -1:

            x_ball, y_ball = self.detect_ball()

            # If ball is detected
            if x_ball != x_ball_old and y_ball != y_ball_old:
                x_ball_old = x_ball
                y_ball_old = y_ball

                # smooth movement
                self.step(self.timeStep * 10)
                if x_ball is not None:
                    p1, r1 = self.predict_movement(x_ball, y_ball, model_key)
                    p2, r2 = self.predict_movement(x_ball, y_ball, model_key)
                    pitch_position, roll_position = (p1 + p2) / 2, (r1 + r2) / 2
                    # move arm if within allowed boundaries
                    if -0.31 < roll_position < 1.3:
                        self.LShoulderR.setPosition(roll_position)
                    if -2 < pitch_position < 2:
                        self.LShoulder.setPosition(pitch_position)


# Initializing Robot class variable
robot = MyRobot(ext_camera_flag=False)
robot.follow_ball(True)
