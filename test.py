from traffic_simulation import *
#import matplotlib.pyplot as plt

seed = input("Please input seed (recommand 33): ")
gamma = input("Please input discount factor gamma (recommand 0.7): ")
alpha = input("Please input learning rate alpha (recommand 0.6): ")
epsilon = input("Please input epsilon (recommand 0.1): ")
episode = input("Please input episode (recommand 50): ")

print("Fix model start simulation!")
num = traffic_fix(int(seed))
print("Fix model finished!")

print("Q-learning model start simulation!")
stopped_list,reward_list = traffic_q(int(seed),float(gamma),float(alpha),float(epsilon),int(episode))
print("Q-learnig model finished!")

#### plot and evaluate
"""
### Fix vs Q-learning
x = [i for i in range(1,len(stopped_list)+1)]
plt.title('Fix model vs Q-learnig model')
plt.plot(x, [num]*len(stopped_list),  color='orange', label='Fix model', linewidth = 2)
plt.plot(x, stopped_list, color='blue', label='Q-learning model', linewidth = 2)
plt.legend(bbox_to_anchor=(0.9,0.65))
plt.xlabel('Episode')
plt.ylabel('The total time of cars stopped in one episode')
plt.show()

### Different parameter
exploration_factor = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
num_list = []
for i in exploration_factor:
    print("exploration_factor is {}".format(i))
    stopped_list,reward_list = traffic_q(int(seed),float(gamma),float(alpha),i,int(episode))
    num_list.append(stopped_list[-1])
plt.title('Q-learnig model with different exploration factor')
plt.bar(range(len(num_list)), num_list, alpha = 0.9, width = 0.4, color='pink', edgecolor='white', tick_label=exploration_factor)
plt.xlabel('Exploration factor')
plt.ylabel('The total time of cars stopped in one episode')
plt.show()
"""
