#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from math import ceil, floor

style.use("ggplot")
# %%
#Environment setting
NB_EPISODES=2500
SIZE=10

#Penalties
HOLE_PENALTY=5
ENEMY_PENALTY=4
GOAL=100
MOVE_PENALTY=0.001

#Epsilon
epsilon=0.8
#epsilon will be updated following: epsilon*=EPISOLON_DECAY
EPSILON_DECAY=0.9 

#Q-table initialization parameter
start_q_table=None #we could have a pickled qtable here too

#Learning rate
LEARNING_RATE=1
DISCOUNT=0.6

#display parameters
SHOW_EVERY=100

#env setting 
start=0
end=SIZE
holes=[3,6]
enemies=[7]
#%%
#Creating the Agent class - the agent is the player of the little game we are creating here
class Blob:
    def __init__(self, x=False):
        if not x:
            self.x = np.random.randint(0, SIZE)
        else: 
            self.x = x

    def __str__(self):
        return f"Location: {self.x}"
    
    def __sub__(self, other):
        return (self.x-other.x)
    
    def move(self, x=False):
        #if there are no values for x or y, move randomly
        if not x:
            self.x+=np.random.randint(0,6)
        else: 
            self.x+=x
        
        #if the player is out of bound put it back at the beginning
        if self.x<0:
            self.x=0
        elif self.x>=SIZE:
            self.x=SIZE-1
     
    def action(self, choice):
        #Gives us 4 total movement options. (0,1,2,3)

        if choice == 0:
            self.move(x=1)
        elif choice == 1:
            self.move(x=-2)
        elif choice == 2:
            self.move(x=3)
        elif choice == 3:
            self.move(x=-1)
        elif choice == 4:
            self.move(x=-2)
        elif choice == 5:
            self.move(x=-3)
        elif choice == 6:
            self.move(x=0)
            
#%%
#Creating the environment - the environment is the universe in which our agent will evolve and with which it will interact
class env: 
    def __init__(self, holes, enemies, start, end):
        self.start=start
        self.end=end
        self.holes=holes
        self.enemies=enemies

    def __str__(self):
        return f"Start: {self.start}\nEnd: {self.end}\nEnemies: {self.enemies}\nHoles: {self.holes}"

#%%
#Initialize the Q-table
if start_q_table is None: 
    q_table={}
    for i in range(0,SIZE+1):
        for j in range(0,SIZE+1):
            q_table[(i,j)]=[np.random.uniform(-20,0) for i in range(7)]
else:
    with open(start_q_table, 'rb') as f:
        q_table=pickle.lead(f)

# %%
episode_rewards = []
for episode in range(NB_EPISODES):
    Mario=Blob(0)
    Level=env(holes, enemies, 0,SIZE)
    Goomba=Blob(Level.enemies[0])

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (Mario.x, Goomba.x)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 6)
        # Take the action!
        Mario.action(action)
        Goomba.move()
        #food.move()

        if Mario.x == Goomba.x:
            reward = -ENEMY_PENALTY
            Mario=Blob(0)
            Goomba=Blob(Level.enemies[0])
        elif Mario.x in Level.holes:
            reward = -HOLE_PENALTY
            Mario=Blob(0)
        else:
            reward = -MOVE_PENALTY

        new_obs = (Mario.x, Goomba.x)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if Mario.x == Level.end:
            print("WON!!")
            new_q = GOAL
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

    #if the goal is reached, end the episode
        episode_reward += reward
        if obs==0:
            break

    #update reward array and epsilon
    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

#%%
#computing the reward's moving average
moving_avg=np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

#plotting the model's performance
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.show()

#saving the Q-table on a .pkl file
with open(f"qtable-{int(time.time())}.pickle","wb") as f:
    pickle.dump(q_table, f)
