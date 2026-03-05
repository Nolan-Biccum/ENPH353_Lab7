import random
import pickle
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        try:
            with open(filename + ".pickle", 'rb') as f:
                self.q = pickle.load(f)
            print("Loaded file: {}".format(filename+".pickle"))
        except FileNotFoundError:
            print("No pickle file found")

        

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename + ".pickle", 'wb') as f:
            pickle.dump(self.q, f)
        
        with open(filename + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["State", "Action", "Value"])
            for (state, action), value in self.q.items():
                writer.writerow([state, action, value])

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 
        if random.random() < self.epsilon:
            action = random.choice(self.actions)

        else:
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)

            count = q_values.count(max_q)
            if count > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q_values[i] == max_q]
                action = random.choice(best_actions)
            else:
                action = self.actions[q_values.index(max_q)]
        
        if return_q:
            return action, self.getQ(state, action)
        return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        old_q = self.getQ(state1, action1)

        next_max_q = max([self.getQ(state2, a) for a in self.actions])

        td_target = reward + self.gamma * next_max_q
        new_q = old_q + self.alpha * (td_target - old_q)

        self.q[(state1, action1)] = new_q

        

        self.q[(state1,action1)] = reward
