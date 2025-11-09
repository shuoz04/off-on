import pygame
import time
import keyboard

# clock = pygame.time.Clock()
# pygame.init()
# joystick = pygame.joystick.Joystick(0)
# joystick.init()
keepPlaying = True
action_set = [
    # (-10, -10), (-10, -5), (-10, 0), (-10, 5), (-10, 10),
    # (-10, -5), (-10, 5),
    (-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
    # (0, -10), (0, -5), (0, 5), (0, 10),
    (5, -10), (5, -5), (5, 0), (5, 5), (5, 10),
    # (10, -10), (10, -5), (10, 0), (10, 5), (10, 10)
    # (10, -5), (10, 5)
]
while keepPlaying:
    # clock.tick(60)
    time.sleep(1)
    # all_action = None
    # pygame.event.get()
        #The zero button is the 'a' button, 1 the 'b' button, 3 the 'y'
        #button, 2 the 'x' button
        # if event.button == 0:
        #     print ("A Has Been Pressed")
        # print(event.button)
        # if event.type==pygame.JOYAXISMOTION:
    # axis = []
    # axis.append(joystick.get_axis(1))
    # # for i in range(4):
    # #     axis.append(joystick.get_axis(i))
    #     # if abs(axis) > 50:
    #     #     print(i)
    # # print(axis)
    # if joystick.get_axis(1) < -0.5:
    #     action = 7
    # elif joystick.get_axis(1) > 0.5:
    #     action = 2
    # else:
    #     action = None
    # if action is not None:
    #     if joystick.get_axis(2) < -0.6:
    #         action -= 2
    #     elif joystick.get_axis(2) < -0.2:
    #         action -= 1
    #     elif joystick.get_axis(2) > 0.6:
    #         action += 2
    #     elif joystick.get_axis(2) > 0.2:
    #         action += 1
    #     all_action = action
    if keyboard.is_pressed('up'):
        action = 7
    elif keyboard.is_pressed('down'):
        action = 2
    else:
        action = None
    if action is not None:
        if keyboard.is_pressed('right'):
            rot = -1
        elif keyboard.is_pressed('left'):
            rot = 1
        else:
            rot = 0
        if keyboard.is_pressed('space'):
            rot *= 2
        action = action + rot
        print(action_set[action])
