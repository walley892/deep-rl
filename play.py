import gym
import numpy as np
import random
import copy
import pickle
from keras.models import load_model
from player import Mind


load = False
render = True


model_path = 'saved.q'


env = gym.make('SpaceInvaders-v0')
print(env.action_space)
action_inverses = {}



def run(m,n = 2, save_after = 10):
    cnt = 1
    rewards = 0    
    for i in range(1,n):
        obs =  env.reset()
        m.reset()
        print('total reward: ', rewards)
        print('average reward: ' + str(rewards/cnt))
        print('lasted: ', cnt)
        rewards = 0
        cnt = 1
        for j in range(1,3000):
            if render:
                env.render()
            cnt+=1
            
            action = m.action(obs)
            
            old_obs = obs
            
            obs, reward, done, info = env.step(action)
            rewards += reward
            m.feedback(action, old_obs, reward/100, obs)
            
            if done:
                break
            

if __name__ == '__main__':
    input_size = env.observation_space.shape
    if load:
        m = Mind(env.action_space.n, input_size,save_path = model_path, load_path = model_path, action_inverses = action_inverses)
    else:
        m = Mind(env.action_space.n, input_size, save_path = model_path, action_inverses = action_inverses)
        
    run(m, 10000)

