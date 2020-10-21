"""tutorial1_tracker controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Keyboard, Display, Motion
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable # storing data while learning
from torch.distributions.distribution import Distribution

class MDN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_gaussians):
        super(MDN, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh()
        )
        
        self.pi = nn.Sequential(
            nn.Linear(n_hidden, n_gaussians),
            nn.Softmax(dim=1)
        )
        
        self.sigma = nn.Linear(n_hidden, n_gaussians)
        self.mu = nn.Linear(n_hidden, n_output*n_gaussians)
        
    def forward(self, x):
        hidden = self.hidden(x)
        pi = self.pi(hidden)
        mu = self.mu(hidden)
        sigma = torch.exp(self.sigma(hidden))
        
        return pi, sigma, mu
     
model = MDN(2,20,2,5)     
    

class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = 32 # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for default
        self.state = 0 # Idle starts for selecting different states
        # Sensors init
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStep)
        self.step(self.timeStep) # Execute one step to get the initial position
        # activate camera
        self.cameraBottom = self.getCamera("CameraBottom")
        self.cameraBottom.enable(2*self.timeStep)
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
        
        self.ChestLED = self.getLED('ChestBoard/Led')
        self.LEarLED = self.getLED('Ears/Led/Left')
        self.REarLED = self.getLED('Ears/Led/Right')
           
        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
        self.currentlyPlaying = False
        self.loadMotionFiles()
        print("done all")  
     
    
    # load motion files
    def loadMotionFiles(self):
        self.handWave = Motion('../../motions/HandWave.motion')
        self.forwards = Motion('../../motions/Forwards50.motion')
        self.backwards = Motion('../../motions/Backwards.motion')
        self.sideStepLeft = Motion('../../motions/SideStepLeft.motion')
        self.sideStepRight = Motion('../../motions/SideStepRight.motion')
        self.turnLeft60 = Motion('../../motions/TurnLeft60.motion')
        self.turnRight60 = Motion('../../motions/TurnRight60.motion')
        self.shoot = Motion('../../motions/shoot.motion')


    def startMotion(self, motion):
        # interrupt current motion
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()

        # start new motion
        motion.play()
        self.currentlyPlaying = motion
        

    # Captures the external camera frames 
    # Returns the image downsampled by 2   
    def camera_read_external(self):
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5) # image downsampled by 2
                        
        return img
            
            
    # Displays the image on the webots camera display interface
    def image_to_display(self, img):
        if self.ext_camera:
            height, width, channels = img.shape
            imageRef = self.displayCamExt.imageNew(cv2.transpose(img).tolist(), Display.RGB, width, height)
            self.displayCamExt.imagePaste(imageRef, 0, 0)
    
    ### TASK 2 ###
    def print_gps(self):
        gps_data = self.gps.getValues();
        print('----------gps----------')
        print(' [x y z] =  [' + str(gps_data[0]) + ',' + str(gps_data[1]) + ',' + str(gps_data[2]) + ']' )
        
        
    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' G for print the gps\n'
            ' UP for walking forward\n'
            ' DOWN for walking backward\n'
            ' LEFT for rotating head to the left\n'
            ' RIGHT for rotating head to the right\n'
            ' W for waving\n'
            ' L for turning left\n'
            ' K for kicking\n' 
            ' S to stop all movements'
            
        )
        
 
    
    def run_keyboard(self):
    
        self.printHelp()
        previous_message = ''

        # Main loop.
        while True:
            # Deal with the pressed keyboard key.
            k = self.keyboard.getKey()
            message = ''
            if k == ord('G'):
                self.print_gps() 
            elif k == ord('H'):
                self.printHelp()
            elif k == Keyboard.UP:
                self.startMotion(self.forwards)
            elif k == Keyboard.DOWN:
                self.startMotion(self.backwards)
            elif k == Keyboard.LEFT:                    
                self.head_yaw.setVelocity(1)             
            elif k == Keyboard.RIGHT:                    
                self.head_yaw.setVelocity(-1)
            elif k == ord('L'):
                self.startMotion(self.turnLeft60)
            elif k == ord('W'):
                self.startMotion(self.handWave)
            elif k == ord('k'):
                self.startMotion(self.shoot)
            elif k == ord('S'):
                if self.currentlyPlaying:
                    self.currentlyPlaying.stop()
                    self.currentlyPlaying = False
                
                self.head_yaw.setVelocity(0)
            
            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break
                
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release() 
            
            
    ### TASK 3 ###      
    def look_at(self, x, y):
        if not (0 < x <1) or not (0 < y < 1): 
            raise ValueError('x and y need to be normalized')
        # Sensibel values between -1.5 and 1.5
        x = (x*3)-1.5
        # Sensibel values between -0.4 and 0.3
        y = (y*0.7)-0.4
        
        self.head_yaw.setVelocity(3)
        self.head_pitch.setVelocity(3)
        self.head_yaw.setPosition(x)
        self.head_pitch.setPosition(y)        
                
    # Face following main function
    def run_face_follower(self):
        # main control loop: perform simulation steps of self.timeStep milliseconds
        # and leave the loop when the simulation is over
        while self.step(self.timeStep) != -1:
        
            cap = cv2.VideoCapture('Recording1.mov')
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # for every frame of the video 
            while(cap.isOpened()):
                ret, frame = cap.read()
                self.step(1)
                # Start simualtion again when video is over
                if ret == False:
                    break
       
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                # if no face detected skip frame
                if len(faces) == 0:
                    continue
                # if two faces are detected skip frame              
                if faces.shape[0] > 1: 
                    continue
                for (x, y, w, h) in faces:
                    # calculate center and normalize
                    face_center = ((x+ w // 2)/width, (y +  h // 2)/height)
                    #move face according to sensible values
                    self.look_at( face_center[0],face_center[1])      
    
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()   
    
    
    ### TASK 4 ###
    def detect_ball(self):
       
       img = self.cameraBottom.getImage()
       height, width = self.cameraBottom.getHeight(), self.cameraBottom.getWidth()
       # turn into np array
       image = np.frombuffer(img, np.uint8).reshape( height, width,4)
       # transform into HSV 
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       # green is defined as tripels between these values
       lower =  np.array([40,50,50])
       upper = np.array([80,255,255])
       # lay a mask over all green values   
       mask = cv2.inRange(image, lower, upper)
       # Image preparation       
       #Erosion
       kernel = np.ones((2,2),np.uint8)
       image = cv2.erode(mask,kernel,iterations = 1)
       #Dilation 
       image= cv2.dilate(image,kernel,iterations = 1)
       # find ball contour
       contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
       if len(contour) != 0:
               contour = np.asarray(contour[0])
               m = cv2.moments(contour)
               # if area of detected blob is reasonable:
               if 3000 > m["m00"] > 1:
                   # calculate momentum
                   cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])   
                   return(cx,cy)  
               else:
                   return (None,None)
       return (None,None)
    
    
    def run_ball_follower(self):      
        yaw_position = 0
        pitch_position = 0
        self.head_yaw.setVelocity(5)
        self.head_pitch.setVelocity(5)        

        while self.step(self.timeStep) != -1:
            x, y = self.detect_ball()
            if x is None:
                continue
            else: 
                # calculate movemnet for yaw and pitch
                K = 0.2
                dx, dy =K*((x/width)-0.5), K*((y/height)-0.5)
                # allowed values between -2 and 2
                # Sensibel values between -1.5 and 1.5
                if -1.5 < yaw_position - dx < 1.5: 
                    yaw_position = yaw_position - dx 
                    self.head_yaw.setPosition(float(yaw_position))
                # allowed values between -0,6 and 0.5
                # Sensibel values between -0.4 and 0.3
                if -0.4 < pitch_position + dy < 0.3: 
                    pitch_position = pitch_position + dy 
                    self.head_pitch.setPosition(float((pitch_position)))
            
            
    ### TASK 5 ###       
    def deactivate(self):
            # stop movemnet, trun of LEDS, move arms in relaxed position
            self.forwards.stop() 
            self.ChestLED.set(int('000000', 16))
            self.LEarLED.set(int('000000', 16))
            self.REarLED.set(int('000000', 16))
            # wait, to simulate human like processing
            self.step(self.timeStep*50)
            # Move arms backwards in order to act taken aback
            self.LShoulder.setPosition(float(1.8))
            self.RShoulder.setPosition(float(1.8))
            self.LShoulderR.setPosition(float(0.3))
            self.RShoulderR.setPosition(float(-0.3))
            self.LElbow.setPosition(float(-1.4))
            self.RElbow.setPosition(float(1.4))
            self.head_yaw.setPosition(float(0))
            self.head_pitch.setPosition(float(0))
            
      
    def run_hri(self):
    # robots walks slowly forward. If a face is detected is indicate recognition and turn to zombie mode. 
    # While in Zombie mode follow face with head movement
    # if a green ball is itroduced to field of vision, show intial shock, then relax and deactivate zombie mode
      while self.step(self.timeStep) != -1:
        
            cap = cv2.VideoCapture('Recording2.mov')
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            Activate = False
           
            while(cap.isOpened()):
                # always move forward to indicate "aliveness"/ activness of NAO 
                self.startMotion(self.forwards)
                self.step(self.timeStep)        
                ret, frame = cap.read()
                # if video ends start over again 
                if ret == False:
                    self.forwards.stop() 
                    break 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                x, y = self.detect_ball()
                # if a ball is detected and Zombie mode activated  
                # deactivate zombie mode  
                if not (x is None) and Activate : 
                    self.deactivate()
                    Activate = False                      
                    while True:
                        # Move arms to relaxed position        
                        self.forwards.stop()
                        self.RElbow.setPosition(0)
                        self.LElbow.setPosition(0)      
                        self.step(self.timeStep*30) 
                        self.startMotion(self.forwards)
                        self.step(self.timeStep*200)
                        # turn on green chest to indicate friendliness 
                        self.ChestLED.set(int('7CFC00', 16))
                        break
                        
                if len(faces) == 0:
                    continue
                # if a face is detected  and Zombie Mode hasn't been activated
                if not Activate: 
                    # turn on side LED to indicate that someone has been spotted
                    self.LEarLED.set(int('a500a5', 16))
                    self.REarLED.set(int('a500a5', 16))
                    self.step(self.timeStep*50)
                    # turn on red chest LED 
                    self.ChestLED.set(int('8b0000', 16))
                    # Move arms stepwise in "Zombie" position
                    self.LShoulder.setPosition(0)
                    self.LElbow.setPosition(0)
                    self.RShoulder.setPosition(0)
                    self.RElbow.setPosition(0)
                    self.step(self.timeStep*50)
                    self.RShoulderR.setPosition(0)
                    self.LShoulderR.setPosition(0)          
                    self.step(self.timeStep*50)
                    Activate = True
              
                # follow face with head to show attentivness
                for (x, y, w, h) in faces:
                    # calculate center and normalize
                    face_center = ((x+ w // 2)/width, (y +  h // 2)/height)
                    self.look_at(face_center[0],face_center[1]) 
            # deactivate once the video is over  
            self.deactivate()  
            self.step(self.timeStep*200)
            
            
    def gather_data(self):

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
                    if -0.31 < roll_position +0.01 < 1.3:
                        roll_position += 0.01
                        self.LShoulderR.setPosition(roll_position)
                if k == Keyboard.LEFT:
                    if -0.31 < roll_position -0.01 < 1.3: 
                        roll_position -= 0.01
                        self.LShoulderR.setPosition(roll_position)
                if k == Keyboard.DOWN:
                    if -2 < pitch_position +0.01 < 2:
                        pitch_position += 0.01
                        self.LShoulder.setPosition(pitch_position)
                if k == Keyboard.UP:
                    if -2 < pitch_position -0.01 < 2: 
                        pitch_position -= 0.01
                        self.LShoulder.setPosition(pitch_position)
                if k == ord('M'): 
                    self.step(400)
                    i += 1
                    print(i) 
                    print(pitch_position, roll_position) 
                    x,y =  self.detect_ball()
                    if x != None: 
                        records.append ([[pitch_position, roll_position], [x,y]])
                    print(self.detect_ball()) 
                if k == ord('P'): 
                    self.step(400)
                    print(records)  
                    records = np.asarray(records) 
                    print(records)
                    np.save("/Users/frederikesmac/Desktop/HRI/data.np", records) 
                if self.step(self.timeStep) == -1:
                    break
            return
             
             
             
    def predict_movement(self,x,y):
        # load model 
        #model.load_state_dict(torch.load('mdn_model'))
        model = torch.load('model.pth')
        model.eval()
        print(model)
        #model
        #... normalize input 
        x_mean = np.load("input_mean.npy") 
        x_std = np.load("input_std.npy")
        target_mean = np.load("target_mean.npy") 
        target_std = np.load("target_std.npy")
        
        input_data = torch.from_numpy(np.empty((1,2)))
        input_data[0,0] = (x-x_mean[0])/x_std[0]
        input_data[0,1] = (y-x_mean[1])/x_std[1]
        #for idx,sample in enumerate(x):
        #  input_data[idx,0] = (sample-x_mean[0])/x_std[0]
        #for idx,sample in enumerate(y):
        #  input_data[idx,1] = (sample-x_mean[0])/x_std[0]
        # calculate arm movemnt 
        pi_variable, sigma_variable, mu_variable = model.forward(input_data)
        y_pred = sample_pred(pi_variable, sigma_variable, mu_variable)
        # de-normalize 
        prediction = (y_pred*target_std)+target_mean
        print(prediction)
        return prediction[0], prediction[1]
        
        
   
    def sample_pred(pi, sigma, mu):
        N, K = pi.shape # Number of samples and number of Gaussians (Same Pi is assumend to work for all Noutputs)
        _, KT = mu.shape # Noutput x Number of Gaussians
        NO = int(KT / K) # Noutput
        pred = Variable(torch.zeros(N, NO))  # sample for all variables
        
        for i in range(N): # loop over samples
            r =  np.random.uniform(0,1) # sample random number from [0, 1) uniform distribution
            summed_pi = 0
            for idx,pi_value in enumerate(pi[i]):
              summed_pi += pi_value.item()
              if summed_pi > r:
                  segment = idx

            for t in range(NO):
                val = np.random.normal(mu[i,t + segment*NO], sigma[i,segment].item())
                pred[i,t] = val
        return pred

           
    def follow_ball(self):
    
        while self.step(self.timeStep) != -1:

             self.LElbow.setPosition(0)
             self.LShoulderR.setVelocity(1)
             self.LShoulder.setVelocity(1) 
             x_ball, y_ball = self.detect_ball()
             if x_ball != None:
                 print(x_ball)
                 roll_position,pitch_position = self.predict_movement(x_ball,y_ball)
                 if -0.31 < roll_position < 1.3:
                    self.LShoulderR.setPosition(roll_position)
                 if -2 < pitch_position < 2:
                    self.LShoulder.setPosition(roll_position)
                 
         
#def learn_position():                 
      
        #self.head_yaw.setVelocity(5)
        #self.head_pitch.setVelocity(5)
        #self.head_pitch.setPosition(0.3)
        #self.head_yaw.setPosition(0.7)
 #   print("here")
  #  supervisor = Supervisor()

        # do this once only
   # ball_node = supervisor.getFromDef("ball candy")
   
        
    #while supervisor.step(TIME_STEP) != -1:
            # this is done repeatedly
      #  values = trans_field.getSFVec3f()
     #   print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
    
#supervisor = Supervisor()
#ball_node = supervisor.getFromDef("ball candy")
   
# create the Robot instance and run the controller
robot = MyRobot(ext_camera_flag = False)
robot.follow_ball()

#robot.run_face_follower()
#robot.run_ball_follower()
#robot.run_hri()


