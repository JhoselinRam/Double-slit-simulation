import numpy as np
import matplotlib as mtl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from wave2d import *

def main():
#--------------------------------- Editable parameters ---------------------------------
    T      = 4.5               #Total time of simultaion in seconds
    width  = 2                 #Width of the domain in meters
    height = 1.0               #Height of the domain in meters
    dt     = 0.001             #Time increment in seconds
    speed  = 0.9               #Speed of the wave in meters/seconds 

    pulseAmplitude = 0.05      #Amplitude of the pulse of the boundary condition
    pulseFrequency = 45.0      #Frequency of the pulse of the boundary condition
    pulseDuration  = round(pulseFrequency*0.5625/speed)/pulseFrequency   #Duration of the pulse of the boundary condition. If negative, the pulse persist the entire simulation


#Slit parameters expressed in proportion of the width and height of the domain [0,1]
    slitXPosition  = 0.375      #'x' position of the double slit
    slitYPosition  = 0.5        #'y' position of the center of the double slit
    slitSeparation = 0.05       #Distance between the slits
    slitThickness  = 0.001      #Thickness of the double slit
    slitAperture   = 0.006      #Aperture of the slits

    colorCap = 0.195        #Proportion of the wave maximum value at the screen with respect to the pulse amplitude, used to trim the color map and the 'y' axis in the screen graph

#Screen parameters expressed in proportion of the width and height of the domain [0,1]
    screenXPosition = 0.7       #'x' position of the screen
    screenYPosition = 0.5       #'y' position of the screen
    screenSize      = 0.8       #Size of the screen

    predictionOrder = 5         #Order of the interference lines prediction, will be displayed 2*order+1 lines
#---------------------------------------------------------------------------------------

#------------------------------- Non editable parameters -------------------------------
    xNodes = int(width/(np.sqrt(2)*speed*dt)) - 1       #Number of nodes in the 'x' direction. Uses the stability condition to avoid "blow to infinity" scenarios
    yNodes = int(height/(np.sqrt(2)*speed*dt)) - 1      #Number of nodes in the 'y' direction. Uses the stability condition to avoid "blow to infinity" scenarios
    dx = width/xNodes       #Increment in the 'x' direction in meters
    dy = height/yNodes      #Increment in the 'y' direction in meters
#---------------------------------------------------------------------------------------

#---------------------------------------- Setup ----------------------------------------
    wave = wave2d(xNodes, yNodes, dx, dy, dt, speed)                #Object of the wave2d class
    wave.setScreenAt(screenSize, screenXPosition, screenYPosition)  #Set the screen in the domain
    wave.setBoundaryConditions(boundary)                            #Set the boundary conditions function

    setInitialConditions(wave, pulseAmplitude, pulseFrequency)                                      #Set the initial conditions
    setDoubleSlit(wave, slitXPosition, slitYPosition, slitSeparation, slitThickness, slitAperture)  #Set the double slit as an obstacle

    mtl.rcParams['toolbar'] = 'None'                                                    #Removes the toolbar of the figure
    figure, (axis1, axis2)  = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 3]})  #Figure and axis for the screen and field
    figure.set_size_inches(19.5555555,11, forward=True)                                 #Set the widht and height of the figure in inches
    axis1.grid()                                                                        #Set the grid for the screen graph
    xScreen = np.linspace(height*(screenYPosition-screenSize/2), height*(screenYPosition+screenSize/2), wave.screen.size)   #Auxiliar variable to plot the initial value in the screen
    yScreen = np.zeros(wave.screen.size)                                                                                    #Auxiliar variable to plot the initial value in the screen
    screenGraph, = axis1.plot(xScreen,yScreen,"b-")                                                                         #Draw the initial value in the screen (zero)

    axis1.set_ylim(-0.0001, colorCap*pulseAmplitude)                                                #Set the 'y' limit of the screen graph
    axis1.set_xlim(height*(screenYPosition-screenSize/2), height*(screenYPosition+screenSize/2))    #Set the 'x' limit of the screen graph
    axis1.set_title("Double slit interference pattern", fontsize=22)                                #Set the title of the screen graph
    axis1.set_xlabel("Distance (m)")                                                                #Set the 'x' label of the screen graph 
    axis1.set_ylabel(r"$|\Psi|$")                                                                   #Set the 'y' label of the screen graph 
    axis2.set_xlabel("Distance (m)")                                                                #Set the 'x' label of the wave graph
    axis2.set_ylabel("Distance (m)")                                                                #Set the 'y' label of the wave graph

    #Draws the theoretical prediction lines in the screen graph
    predictionGraph = setPredictionOnScreen(axis1, predictionOrder, pulseFrequency, slitSeparation*height, slitAperture*height, (screenXPosition-slitXPosition)*width, speed, colorCap*pulseAmplitude, height)
    axis1.legend((screenGraph, predictionGraph), (r"$\max(|\Psi|)$", "Predicted fringe"))   #Set the legends of the screen graph
    plt.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.045, hspace=0.1)          #Adjust the spacing of the graphs within the figure
    figure.patch.set_facecolor("0.35")                                                      #Set the background color of the figure into gray


    fieldGraph = axis2.imshow(wave.getWave().T, extent=(0,width,0,height),cmap=plt.get_cmap('cividis'), aspect="equal", vmin=-colorCap*pulseAmplitude, vmax=colorCap*pulseAmplitude)    #Graph of the wave

    #Black magic animation thingy
    simulation = animation.FuncAnimation(figure, photogram,             #Figure and the "update function"
                                                frames=int(T/dt),       #Amount of frames
                                                interval=1000*dt,       #Interval between each frame, used in the save method
                                                repeat=False,           #Dont repeat the animation after it ends
                                                blit=True,              #To optimize the drawin
                                                fargs=(wave,screenGraph,fieldGraph,pulseDuration,pulseAmplitude,pulseFrequency,xScreen, T)) #Arguments of the "update function"
    

    #writeGif = animation.FFMpegWriter(fps=60)                      #Creates the writer to save the video, FFMPEG must be installed in the computer
    #simulation.save("doubleSlit02.mp4", writer=writeGif, dpi=100)  #Saves the video
    plt.show()                                                      #Show the animation in screen

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------



#---------------------------------- Auxiliar functions ---------------------------------
def setInitialConditions(wave, amplitude, frequency):
    wave.field[0,:,:] = 0
    wave.field[1,0,:] = amplitude*np.sin(2*np.pi*frequency * wave.dt)   #Set the pulse at the first time step


def setDoubleSlit(wave, xPosition, yPosition, separation, thickness, aperture):
    xNodes, yNodes = wave.field[0,:,:].shape                    #Number of nodes in both dimensions
    slit = np.ones(shape=(xNodes, yNodes), dtype="float64")     #The obstacle must have the same shape as the field

    left, right      = int(xNodes*(xPosition - thickness/2)), int(xNodes*(xPosition + thickness/2))                 #'x' coordinates of the leftmost and rightmost part of the slit
    upperY0, upperY1 = int(yNodes*(yPosition - separation/2)), int(yNodes*(yPosition - separation/2 - aperture))    #'y' coordenates of the start and end of the upper slit
    lowerY0, lowerY1 = int(yNodes*(yPosition + separation/2)), int(yNodes*(yPosition + separation/2 + aperture))    #'y' coordenates of the start and end of the lower slit

    #Set the corresponding areas of the slit to zero
    slit[left:right, :upperY1]        = 0.0
    slit[left:right, upperY0:lowerY0] = 0.0
    slit[left:right, lowerY1:]        = 0.0

    wave.setObstacle(slit)  #Set the slit as an obstacle for the wave


def boundary(step, duration, amplitude, frequency, wave):
    t = wave.dt * (step + 2)    #Time in the simulation, the 'step + 2' part takes into acount the initial condition time
    if duration <= 0:               #If the duration of the pulse its zero or less
        wave.field[2,0,:] = amplitude*np.sin(2*np.pi*frequency*t)   #The pulse last indefinitely
    else:                           #Otherwise
        if t <= duration:               #If the time is less than the pulse duration
           wave.field[2,0,:] = amplitude*np.sin(2*np.pi*frequency*t)    #Compute the pulse 
        else:                           #Else
            wave.field[2,0,:] = 0.0         #Set the wave to zero in this edge

    wave.field[2,-1,:] = 0.0                    #Set the wave to zero in the right edge of the domain
    wave.field[2,:,0] = wave.field[2,:,1]       #Boundary conditions of the upper edge of the domain
    wave.field[2,:,-1] = wave.field[2,:,-2]     #Boundary condition of the lower edge of the domain


def setPredictionOnScreen(axis, order, frequency, separation, aperture, screenDistance, speed, yMax, height):
    yMaxima = np.array([0, yMax])                
    xMaxima = np.array([height/2, height/2])        #Position of the order zero prediction (in the middle of the screen)
    
    graph, = axis.plot(xMaxima, yMaxima, "k--")     #Draws the order zero prediction and store a reference to that plot, used to set the legend in the graph
    for i in range(1, order+1):
        fringe = maximaDistance(i, frequency, separation, aperture, screenDistance, speed)      #Compute the position of the fringe of the order 'n' prediction
        xMaxima = np.array([1, 1])*(fringe + height/2)
        axis.plot(xMaxima, yMaxima, "k--")              #Draw the prediction line

        xMaxima = np.array([1, 1])*(-fringe + height/2) #And the fringe of the order '-n' prediction
        axis.plot(xMaxima, yMaxima, "k--")

    return graph    #Return the reference to the prediction zero plot



def maximaDistance(order, frequency, separation, aperture, screenDistance, speed):
    wavelenght = speed/frequency    #Wavelenght of the wave
#--------------------------------------------------------------------------------
#Usual law to obtain the position of the fringes in the double slith problem
   
    # angle = np.arcsin(order*wavelenght/(separation + aperture))

    # fringe = screenDistance * np.tan(angle)
    
    # print(fringe)
    # return fringe
#--------------------------------------------------------------------------------
#Use instead an exact (more complex) solution.

    A = (separation + aperture)/2
    a = order**2*wavelenght**2/(4*A**2) - 1
    b = order*wavelenght - order**3*wavelenght**3/(4*A**2)
    c = screenDistance**2+order**4*wavelenght**4/(16*A**2) - order**2*wavelenght**2/2 + A**2

    R = (-b-np.sqrt(b**2 - 4*a*c))/(2*a)

    fringe = order*wavelenght/(2*A)*(R - order*wavelenght/2)
    return fringe




def photogram(step,wave,screenGraph,fieldGraph,pulseDuration,pulseAmplitude,pulseFrequency,xScreen, T):
    time = wave.dt * (step+1)               #Time in the simulation
    wave.update(step,pulseDuration,pulseAmplitude,pulseFrequency,wave)  #Update the evolution of the wave in one time step
   
    screenData = wave.getAbsoluteScreen()       #Get the historical maximum of each point in the screen
    fieldGraph.set_data(wave.getWave().T)       #Update the data in the wave graph
    screenGraph.set_data(xScreen, screenData)   #Update the data in the screen graph

    print(f"Simulation time: {time:0.3f}   Max on screen: {np.max(screenData):0.5f}   Completed: {time/T:0.2%}")    #Print some useful information in the terminal

    return fieldGraph, screenGraph,     #Return the graphs









if __name__ == "__main__":
    main()
