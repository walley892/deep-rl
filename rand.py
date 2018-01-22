import gym
import numpy as np
import random
import copy
import pickle
#from keras.models import load_model
#from player import Mind


load = True
render = False


model_path = 'saved.q'


env = gym.make('SpaceInvaders-v0')
print(env.action_space)
action_inverses = {}#{0: 2, 2: 0}



def run(n = 2, save_after = 10):
    cnt = 1
    rewards = 0    
    for i in range(1,n):
        obs =  env.reset()
        print('total reward: ', rewards)
        print('average reward: ' + str(rewards/cnt))
        print('lasted: ', cnt)
        rewards = 0
        cnt = 1
        for j in range(1,3000):
            if render:
                env.render()
            cnt+=1
            
            action = np.random.randint(0,env.action_space.n) 
            old_obs = obs
            
            obs, reward, done, info = env.step(action)
            rewards += reward
            
            if done:
                break
                        

if __name__ == '__main__':
    run(10000)

