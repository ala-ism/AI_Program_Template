#%%
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")
# %%
SIZE=10 #The environment is a 10x10 sized square

NB_EPISODES=25000 #number of games to be played to train our model

MOVE_PENALTY=1 #everytime the agent move one point is removed from its reward

ENEMY_PENALTY=20 #everytime the agent touches an enemy it looses 20 points

FOOD_REWARD=50 #getting to the food gets you 50 points

epsilon=0.5 #the exploration parameter is set to 0.5

EPSILON_DECAY=0.9 #epsilon will be updated following: epsilon*=EPISOLON_DECAY

SHOW_EVERY=100 #We do not want to display information on all the games that are played to save memory and make the training faster

start_q_table=None #we could have a pickled qtable here too

LEARNING_RATE=0.1 #much weight is put on past rather than future information

DISCOUNT=0.9 #the max potential futur reward is highly discounted

#create stakeholders' IDs for the color dictionnary 
PLAYER_ID=1
FOOD_ID=2
ENEMY_ID=3

#colors:
d={1:(255,175,0), 2:(0,255,0), 3:(0,0,255)} #setting up the colors for our visual representation

#%%
#This part consists in creating our agents. To do so, we are creating class objects: 
## In Python, classes can be though of as specific objects composed of various basic other objects and subject to specific functions and operands.
## A famous class example in Python would be the pandas.DataFrame class: it is composed of simpler objects and has its own list of functions
class Blob:
    #The agent is characterized by its position: an (x,y) couple
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        
    #This function is our first operator override: we make sure that a print(agent) would print the two coordinates
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    #This function is the second operator override: substracting two agents amounts to substracting the Xs and the Ys
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    #Defining the set of move that we authorize our agent to make
    def move(self, x=False, y=False):
        #if there are no values for x or y, move randomly
        if not x:
            self.x+=np.random.randint(-1,2)
        else: 
            self.x+=x
        
        if not y:
            self.y +=np.random.randint(-1,2)
        else:
            self.y+=y
        
        #if the player is out of bound, put it back in the closest angle of the board:
        if self.x<0:
            self.x=0
        elif self.x>SIZE-1:
            self.x=SIZE-1
        
        if self.y<0:
            self.y=0
        elif self.y>SIZE-1:
            self.y=SIZE-1
    
    #This functions associates choices to actions
    def action(self, choice):
        #Gives us 4 total movement options. (0,1,2,3)

        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
#%%
#The q-table is randomly initialized
if start_q_table is None: 
    q_table={}
    for i in range(-SIZE+1, SIZE):
        for j in range(-SIZE+1,SIZE):
            for k in range(-SIZE+1,SIZE):
                for l in range(-SIZE+1,SIZE):
                    q_table[((i,j), (k,l))]=[np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q-table, 'rb') as f:
        q_table=pickle.lead(f)
# %%
#The game is played 
episode_rewards = []

for episode in range(NB_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

 
        enemy.move()
        #food.move()


        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
     
        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_ID]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_ID]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_ID]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # display it
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

#%%
moving_avg=np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.show()

with open(f"qtable-{int(time.time())}.pickle","wb") as f:
    pickle.dump(q_table, f)
