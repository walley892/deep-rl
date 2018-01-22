from networks import FCNN, CNN
import numpy as np
import pickle


class Mind:
    def __init__(self, n_actions, input_shape, save_path = None, load_path= None, action_inverses = {}, update_interval = 256*2, save_interval = 5):
        self.n_actions = n_actions
        self.save_path = save_path
        self.network = CNN(n_out = n_actions, input_shape= input_shape)
        
        if load_path != None:
            self.network.load(load_path)
        
        self.n_features = input_shape
        self.data = []
        self.current_episode_count = 1

        self.random_actions = 0

        self.last_action = None
        self.last_action_random = False

        self.action_inverses = action_inverses

        self.lifetime = 1
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.n_updates = 1

    def q(self, state):
        return self.network.predict(np.expand_dims(np.array(state), axis = 0))[0]

    def should_explore(self, state):
        if np.random.random() < 1000/(1000+self.lifetime):
            return True
        return False
    
    def explore_action(self, state):
        return np.random.randint(0, self.n_actions)
    
    def action(self, state):
        q = self.q(state)
#        if self.last_action_random:
#            if self.last_action in self.action_inverses:
#                q[self.action_inverses[self.last_action]] = float('-inf')

        action = np.argmax(q)
        
        if self.should_explore(state):
            self.random_actions +=1
            action = self.explore_action(state)
            self.last_action_random = True
        else:
            self.last_action_random = False

        if self.lifetime % self.update_interval == 0:
            self.update(alpha = 0.9)
            self.n_updates +=1
            if self.n_updates % self.save_interval == 0:
                if self.save_path != None:
                    self.save(self.save_path)
                    print('saved')

        self.last_action = action
        self.current_episode_count += 1
        self.lifetime += 1

        return action

    def save(self, path):
        self.network.save(path) 

    
    def reset(self):
        self.count = 1
        print('Random actions: ', self.random_actions)
        self.random_actions = 0
    def q_target(self, reward, best_next, alpha):
        return reward + alpha * best_next
    def feedback(self, old_action, old_state, reward, new_state):
        self.data.append({'Q_max': np.max(self.q(new_state)),'reward':reward, 'old_state': old_state, 'old_action':old_action})
    
    def update(self,  alpha = 0.6):
        np.random.shuffle(self.data)
        samples = self.data
        self.data = []
        states = []
        ys = []
        
        for sample in samples:
            y = self.q(sample['old_state'])
            y[sample['old_action']]  = self.q_target(sample['reward'], best_next = sample['Q_max'], alpha = alpha)
            #y[sample['old_action']]  = sarsa_target(sample['reward'], next_action = sample['Q_max'], alpha = alpha)
             
            
            states.append(sample['old_state'])

            ys.append(y)
        self.network.train(np.array(states), np.array(ys))
