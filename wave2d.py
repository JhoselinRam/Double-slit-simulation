import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.twodim_base import mask_indices

class wave2d():
    def __init__(self, width, height, dx, dy, dt, speed):
        self.widht  = width     #Lenght of the domain in the 'x' direction in meters
        self.height = height    #lenght of the domain in the 'y' direction in meters
        self.dx     = dx        #Size of the spatial increment in the 'x' direction in meters
        self.dy     = dy        #Size of the spatial increment in the 'y' direction in meters
        self.dt     = dt        #Size of the temporal increment in seconds
        self.speed  = speed     #Speed of the wave in meters/second

        self.field     = np.zeros(shape=(3, int(width/dx), int(height/dy)), dtype="float64")     #The actual wave, solution of the equation
        self.obstacle  = np.ones(shape=(self.field.shape), dtype="float64")                      #Obstacle inside the domain.
        self.mask      = self.obstacle.copy()                                                    #Same as the obstacle, needed for visualisation
        self._boundary = None                                                                    #No default boundary conditions. This must be a function given by the user

        self.screenOn   = False   #No default screen
        self.screen     = None
        self.screenXpos = None
        self.screenYpos = None
        self.screenAlignment = None

    
    def setBoundaryConditions(self, func):
        self._boundary = func   #Set the boundary condition function defined by the user


    def setObstacle(self, obstacle):
        if self.field.shape == obstacle.shape and obstacle.dtype == "float64":  #The obstacle must be the same shape and type as the field for this to work
            self.obstacle = obstacle.copy()          #Copy the obstacle
            self.mask[self.obstacle == 0] = np.nan   #Set the mask to nan anywere the obstacle is


    def setScreenAt(self, size, xPosition, yPosition, alignment=0):
        self.screenOn   = True                                        #Set the use of the screen
        self.screenXpos = int((self.widht/self.dx - 1)*xPosition)     #Set the 'x' position of the screen center, xPosition is relative to the total width of the domain [0,1]
        self.screenYpos = int((self.height/self.dy - 1)*yPosition)    #Set the 'y' position of the screen center, yPosition is relative to the total height of the domain [0,1]
        self.screenAlignment = alignment                              #0 - Vertical screen.   1 - Horizontal screen

        if(alignment == 0): #Set the absolute size of the screeen given the alignment [0,1]
            self.screen = np.zeros(shape=(int(self.height/self.dy*size)), dtype="float64")
        elif(alignment == 1):
            self.screen = np.zeros(shape=(int(self.widht/self.dx*size)), dtype="float64")


    def getScreen(self):
        if self.screenOn == True:   #If the screen is in use return its value depending on alignment
            if self.screenAlignment == 0:               
                screenStart = int(self.screenYpos - self.screen.size/2)   #Auxiliar variable, use to not overshoot the screen size
                self.screen = self.field[2, self.screenXpos, screenStart:screenStart+self.screen.size]  #Take the value of the field in the screens position
            elif self.screenAlignment == 1:
                screenStart = int(self.screenXpos - self.screen.size/2)   #Auxiliar variable, use to not overshoot the screen size
                self.screen = self.field[2, screenStart:screenStart+self.screen.size, self.screenYpos]  #Take que value of the field in the screens position
            
            return self.screen.copy()   #Retutn a copy of the screen


    def getWave(self):
        wave = self.field[2,:,:]     #Auxiliar variable, use for not destructive visualization of the field

        if self.screenOn == True:
            if self.screenAlignment == 0:
                screenStart = int(self.screenYpos - self.screen.size/2)   #Auxiliar variable, use to not overshoot the screen size
                wave[self.screenXpos, screenStart:screenStart+self.screen.size] = np.nan    #Set the wave to nan in the screen position for visualization purposes
            elif self.screenAlignment == 1:
                screenStart = int(self.screenXpos - self.screen.size/2)   #Auxiliar variable, use to not overshoot the screen size
                wave[screenStart:screenStart+self.screen.size, self.screenYpos] = np.nan    #Set the wave to nan in the screen position for visualization purposes
        
        wave *= self.mask   #Set the wave to nan in the obstacle location for visualization purposes
        return wave         #Return the wave

    def update(self, *boundaryArgs):
            #Update the wave in the next dt using finite differences
        self.field[2, 1:-1, 1:-1] = (self.dt*self.speed/self.dx)**2 * (self.field[1, 2:, 1:-1] + self.field[1, :-2, 1:-1] - 2*self.field[1, 1:-1, 1:-1]) +\
                                    (self.dt*self.speed/self.dy)**2 * (self.field[1, 1:-1, 2:] + self.field[1, 1:-1, :-2] - 2*self.field[1, 1:-1, 1:-1]) +\
                                    2*self.field[1, 1:-1, 1:-1] - self.field[0, 1:-1, 1:-1]
        
        self.field[2,:,:] *= self.obstacle      #Aplies the obstacle condition
        self._boundary(*boundaryArgs)           #Aplies the boundary condition
        self.field[0,:,:] = self.field[1,:,:]   #Update the last time interval for the next iteration
        self.field[1,:,:] = self.field[2,:,:]   #Update the second last time interval for the next iteration



