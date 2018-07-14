"""
Simulation region:
1000*1000

Cars:
Cars enter two roads at random intervals.
Cars move one unit at each time-step. (speed = 10 pixels/time_step)
Cars travel only one direction.
Cars can not at same space.
Cars will be removed when they reach the boundary of simulation region.
Cars queue behind stopped cars.

Roads:
On two intersecting roads: left come road = 1; up come road = 2.
Road length = 100 units.
1 unit = 10 pixels.

Controller:
Controller change lights,after at least 3 time-steps since the last change.

Traffic lights:
Controlled by two traffic lights: Green = [0,255,0]; Red = [0,0,255].

Performance measure:
Sum of time-steps that the number of cars queued in a period of time (1000 time-steps).
For example:
car_stop_time=[1,2,3,4,5] means there are 5 cars stopped in 1000 time_steps, which are 1,2,3,4,5 represented.
"""

from math import *
from random import *
import cv2
import numpy as np
import copy

#### Simulate the traffic generation, flow and traffic lights
class Car(object):
    def __init__(self, road = None, arrival_time = inf, wait_time = 0, position = [[inf,inf],[inf,inf]]):
        self.road = road
        self.arrival_time = arrival_time
        self.wait_time = wait_time
        self.position = position

class Light(object):
    def __init__(self, road = None, color = None, start_time = inf, end_time = inf):
        self.road = road
        self.color = color
        self.start_time = start_time
        self.end_time = end_time

################################### Fix change time simulation ####################################
def traffic_fix(set_seed):
    ### set seed
    seed(set_seed)

    ### Inintial time
    master_clock = 0

    ### Initial car list
    car_list_left = []
    car_list_up = []

    ### Initial two lights
    light_left = Light(road = 1, color = [0,255,0], start_time = 0, end_time = 10)
    light_up = Light(road = 2, color = [0,0,255], start_time = 0, end_time = 10)
    change_time = 10

    ### Initial original image
    org_img = np.zeros((1000,1000,3), np.uint8)
    org_img = org_img + 125
    # roads
    org_img[490:510, :] = [0,0,0]
    org_img[:, 490:510] = [0,0,0]
    # lights
    org_img[490:500,480:490] = light_left.color
    org_img[480:490,500:510] = light_up.color

    # the sum of the number of time-steps the number of cars is queued in a period of time
    num = 0
    while master_clock <= 1000:

        ### Change lights
        if light_left.end_time == master_clock:
            light_left.start_time = master_clock
            light_left.end_time = master_clock + change_time
            if light_left.color == [0,255,0]:
                light_left.color = [0,0,255]
            else:
                light_left.color = [0,255,0]
        if light_up.end_time == master_clock:
            light_up.start_time = master_clock
            light_up.end_time = master_clock + change_time
            if light_up.color == [0,255,0]:
                light_up.color = [0,0,255]
            else:
                light_up.color = [0,255,0]

        ### Move cars
        ## Cars from left
        if len(car_list_left) > 0:
            # first car out of boundary
            if car_list_left[0].position[1][1] == 1000:
                car_list_left = car_list_left[1:]
            for i in range(0,len(car_list_left)):
                # car reach to light
                if np.array_equal(car_list_left[i].position[1], [480,490]):
                    # if green light
                    if np.array_equal(light_left.color, [0,255,0]):
                        car_list_left[i].position[1] = [car_list_left[i].position[1][0]+10,car_list_left[i].position[1][1]+10]
                        continue
                    # if red light
                    else:
                        num+=1
                        continue
                # other cars
                if not np.array_equal([car_list_left[i].position[1][0]+10,car_list_left[i].position[1][1]+10], car_list_left[i-1].position[1]):
                    car_list_left[i].position[1] = [car_list_left[i].position[1][0]+10,car_list_left[i].position[1][1]+10]
                else:
                    num+=1
        ## Cars from up
        if len(car_list_up) > 0:
            if car_list_up[0].position[0][1] == 1000:
                car_list_up = car_list_up[1:]
            for i in range(0,len(car_list_up)):
                crash_flag = False
                if np.array_equal(car_list_up[i].position[0], [480,490]):
                    if np.array_equal(light_up.color, [0,255,0]):
                        for j in car_list_left:
                            if np.array_equal(j.position[1],[500,510]):
                                crash_flag = True
                                break
                        if not crash_flag:
                            car_list_up[i].position[0] = [car_list_up[i].position[0][0]+10,car_list_up[i].position[0][1]+10]
                            continue
                        else:
                            num+=1
                            continue
                    else:
                        num+=1
                        #print("u:{}".format(master_clock))
                        continue
                if not np.array_equal([car_list_up[i].position[0][0]+10,car_list_up[i].position[0][1]+10], car_list_up[i-1].position[0]):
                    car_list_up[i].position[0] = [car_list_up[i].position[0][0]+10,car_list_up[i].position[0][1]+10]
                else:
                    num+=1

        ### Generate new car
        ## Cars from left
        if master_clock % (randint(1,10)) == 0:
            new_car = Car(road = 1, arrival_time = master_clock, position = [[490,500],[0,10]])
            car_list_left.append(new_car)
        ## Cars from up
        if master_clock % (randint(1,10)) == 0:
            new_car = Car(road = 2, arrival_time = master_clock, position = [[0,10],[500,510]])
            car_list_up.append(new_car)

        ### Write in new img
        new_img = org_img.copy()
        new_img[490:500,480:490] = light_left.color
        new_img[480:490,500:510] = light_up.color
        for car in car_list_left:
            new_img[car.position[0][0]+1:car.position[0][1]-1, car.position[1][0]+1:car.position[1][1]-1] = [255,255,255]
        for car in car_list_up:
            new_img[car.position[0][0]+1:car.position[0][1]-1, car.position[1][0]+1:car.position[1][1]-1] = [255,255,255]
        cv2.imwrite('./images_fix/'+str(master_clock)+'.jpg',new_img)

        master_clock += 1

    #### Display the simualtion on the screen
    video_dir = './fix.avi'
    fps = 10
    numm = 1001
    img_size = (1000,1000)
    fourcc  = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in range(0, numm):
        frame = cv2.imread('./images_fix/'+str(i)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()


    print("The total time of cars stopped in 1000 time-steps is: {}".format(num))
    return num

########################################## Q learning traffic simulation ###########################################
### Select next action
def select_action_now(Q_table, state_now, actions, epsilon,
                      light_left, light_up, master_clock):
    if state_now[3] < 3:
        # can not change light
        action_now = -1
    else:
        #can change light
        if (light_left.end_time == master_clock) and (light_up.end_time == master_clock):
            return 10
        if random() < epsilon:
            action_now = choice(actions)
            return action_now
        else:
            if ((state_now,0) not in Q_table) and ((state_now,1) not in Q_table):
                return -1
            q_list = []
            for action in actions:
                if len(Q_table) == 0:
                    q_list.append(0.0)
                else:
                    if (state_now,action) in Q_table:
                        q_list.append(Q_table[(state_now,action)])
                    else:
                        q_list.append(0.0)
            maxQ = max(q_list)
            count = q_list.count(maxQ)
            if count > 1:
                action_now = choice(actions)
            else:
                action_now = actions[q_list.index(maxQ)]
    return action_now

### Estimate of optimal future value
def max_Q_next(Q_table, state_next, actions):
    q_list = []
    for action in actions:
        if len(Q_table) == 0:
            q_list.append(0.0)
        else:
            if (state_next,action) in Q_table:
                q_list.append(Q_table[(state_next,action)])
            else:
                q_list.append(0.0)
    maxQ = max(q_list)
    return maxQ

### Start simulation
def traffic_q(set_seed, gamma, alpha, epsilon, episode):
    ### Initialize Q_table
    Q_table = {}
    #end_learning = 0
    #last_result = {}

    ### Initialize R_table: all states in R_table reward -1
    r_table = []
    r = [0]
    for i in range(0,10):
        r.append(i)
        r.append(1)
        for j in range(0,4):
            r.append(j)
            r_table.append(r)
            r = r [:-1]
        r = [0]
    r = []
    for i in range(0,10):
        r.append(i)
        r.append(0)
        r.append(0)
        for j in range(0,4):
            r.append(j)
            r_table.append(r)
            r = r [:-1]
        r = []
    R_table = []
    for r in r_table:
        if r not in R_table:
            R_table.append(r)


    ### Initialize actions: (switch 1, not switch 0)
    actions = [0,1]

    ### Initialize parameters
    """
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    episode = 50
    """

    num_list = []
    reward_list = []
    for e in range(0, episode):
        ### set seed
        seed(set_seed)

        ### Inintial time
        master_clock = 0

        ### Initial car list
        car_list_left = []
        car_list_up = []

        ### Initial two lights
        light_left = Light(road = 1, color = [0,255,0], start_time = 0, end_time = 10)
        light_up = Light(road = 2, color = [0,0,255], start_time = 0, end_time = 10)

        ### Initial state_now
        state_now = (9,9,0,0)
        reward = 0

        ### Initial number of car stopped
        num = 0

        if e == episode-1:
            ### Initial original image
            org_img = np.zeros((1000,1000,3), np.uint8)
            org_img = org_img + 125
            # roads
            org_img[490:510, :] = [0,0,0]
            org_img[:, 490:510] = [0,0,0]
            # lights
            org_img[490:500,480:490] = light_left.color
            org_img[480:490,500:510] = light_up.color

        while master_clock <= 1000:
            ### Change lights
            ## Choose an action base on epsilon greedy
            action_now = select_action_now(Q_table, state_now, actions, epsilon,
                                           light_left, light_up, master_clock)
            if action_now>0:
                if light_left.color == [0,255,0]:
                    light_left.color = [0,0,255]
                    light_state = 1
                else:
                    light_left.color = [0,255,0]
                    light_state = 0

                if light_up.color == [0,255,0]:
                    light_up.color = [0,0,255]
                    light_state = 0
                else:
                    light_up.color = [0,255,0]
                    light_state = 1
                light_left.start_time = master_clock
                light_left.end_time = master_clock + 10
                light_up.start_time = master_clock
                light_up.end_time = master_clock + 10
                light_delay = 0
            else:
                if light_left.color == [0,255,0]:
                    light_state = 0
                else:
                    light_state = 1
                if light_up.color == [0,255,0]:
                    light_state = 1
                else:
                    light_state = 0

                if state_now[3] < 3:
                    light_delay = state_now[3] + 1

            ### Move cars
            ## Cars from left
            if len(car_list_left) > 0:
                # first car out of boundary
                if car_list_left[0].position[1][1] == 1000:
                    car_list_left = car_list_left[1:]
                for i in range(0,len(car_list_left)):
                    # car reach to light
                    if np.array_equal(car_list_left[i].position[1], [480,490]):
                        # if green light
                        if np.array_equal(light_left.color, [0,255,0]):
                            car_list_left[i].position[1] = [car_list_left[i].position[1][0]+10,car_list_left[i].position[1][1]+10]
                            continue
                        # if red light
                        else:
                            num+=1
                            continue
                    # other cars
                    if not np.array_equal([car_list_left[i].position[1][0]+10,car_list_left[i].position[1][1]+10], car_list_left[i-1].position[1]):
                        car_list_left[i].position[1] = [car_list_left[i].position[1][0]+10,car_list_left[i].position[1][1]+10]
                    else:
                        num+=1
            ## Cars from up
            if len(car_list_up) > 0:
                if car_list_up[0].position[0][1] == 1000:
                    car_list_up = car_list_up[1:]
                for i in range(0,len(car_list_up)):
                    crash_flag = False
                    if np.array_equal(car_list_up[i].position[0], [480,490]):
                        if np.array_equal(light_up.color, [0,255,0]):
                            for j in car_list_left:
                                if np.array_equal(j.position[1],[500,510]):
                                    crash_flag = True
                                    break
                            if not crash_flag:
                                car_list_up[i].position[0] = [car_list_up[i].position[0][0]+10,car_list_up[i].position[0][1]+10]
                                continue
                            else:
                                num+=1
                                continue
                        else:
                            num+=1
                            continue
                    if not np.array_equal([car_list_up[i].position[0][0]+10,car_list_up[i].position[0][1]+10], car_list_up[i-1].position[0]):
                        car_list_up[i].position[0] = [car_list_up[i].position[0][0]+10,car_list_up[i].position[0][1]+10]
                    else:
                        num+=1

            ### Generate new car
            ## Cars from left
            if master_clock % (randint(1,10)) == 0:
                new_car = Car(road = 1, arrival_time = master_clock, position = [[490,500],[0,10]])
                car_list_left.append(new_car)
            ## Cars from up
            if master_clock % (randint(1,10)) == 0:
                new_car = Car(road = 2, arrival_time = master_clock, position = [[0,10],[500,510]])
                car_list_up.append(new_car)

            ### Closest car
            car_closest_left = []
            car_closest_up = []
            ## Car from left
            for car in car_list_left:
                if car.position[1][1] <= 490 and car.position[1][1] >= 410:
                    car_closest_left.append((490-car.position[1][1])/10)
            ## Car from up
            for car in car_list_up:
                if car.position[0][1] <= 490 and car.position[0][1] >= 410:
                    car_closest_up.append((490-car.position[0][1])/10)


            ### Q-Learning

            ### State_next
            if len(car_closest_left) == 0:
                if len(car_closest_up) == 0:
                    state_next = (9,9,light_state,light_delay)
                else:
                    state_next = (9,min(car_closest_up),light_state,light_delay)
            else:
                if len(car_closest_up) == 0:
                    state_next = (min(car_closest_left),9,light_state,light_delay)
                else:
                    state_next = (min(car_closest_left),min(car_closest_up),light_state,light_delay)

            ### Take the action and observe the reward as well as the new state
            if len(Q_table) == 0:
                q_now = 0.0
            else:
                if (state_now,action_now) in Q_table:
                    q_now = Q_table[(state_now,action_now)]
                else:
                    q_now = 0.0

            if list(state_now) in R_table:
                reward_now = -1.0
            else:
                reward_now = 0.0
            reward += reward_now

            if(action_now==1) or (action_now==0):
                max_q_next = max_Q_next(Q_table, state_next, actions)

                ## Update the value in Q table using the observed reward and the maximum reward for next state
                q_update = (1.0-alpha)*q_now + alpha*(reward_now+gamma*max_q_next)
                Q_table.update({(state_now,action_now):q_update})

            ## Update state
            state_now = copy.deepcopy(state_next)

            ## Write image for one time step
            if e == episode-1:
                ### Write in new img
                new_img = org_img.copy()
                new_img[490:500,480:490] = light_left.color
                new_img[480:490,500:510] = light_up.color
                for car in car_list_left:
                    new_img[car.position[0][0]+1:car.position[0][1]-1, car.position[1][0]+1:car.position[1][1]-1] = [255,255,255]
                for car in car_list_up:
                    new_img[car.position[0][0]+1:car.position[0][1]-1, car.position[1][0]+1:car.position[1][1]-1] = [255,255,255]
                cv2.imwrite('./images_q/'+str(master_clock)+'.jpg',new_img)

            ## Update master_clock
            master_clock += 1


        print("episode:{}   The total time of cars stopped in this episode is:{}".format(e,num))
        #print("episode:{}   The rewards in this episode is:{}".format(e,reward))
        num_list.append(num)
        reward_list.append(reward)
        """
        ## end learning if convergence
        if last_result == num:
            end_learning += 1
            if end_learning > 20:
                return(num_list,reward_list)
        else:
            last_result = num
            end_learning = 0
        """

    #### Display the simualtion on the screen
    video_dir = './q.avi'
    fps = 10
    numm = 1001
    img_size = (1000,1000)
    fourcc  = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in range(0, numm):
        frame = cv2.imread('./images_q/'+str(i)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()

    ## final results
    return(num_list,reward_list)
