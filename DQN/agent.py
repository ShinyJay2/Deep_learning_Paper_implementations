import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DQN
import numpy as np


# Let's define our Agent
# Since we don't require Module, we don't have to inherit DQNAgent from nn.Module.

class DQNAgent(nn.Module):
    def __init__(self,input_shape, n_actions):
        super().__init__()

        # DQN tries to approximate Q-function through Neurl Networks (outputs)
        # Agent choose action based on these Q values

        self.policy_net = DQN(input_shape, n_actions)

        ## IMPORTANT !!
        ### The idea is that if we use the same network to both predict the current Q-values and estimate the future Q-values, 
        # the network updates can become highly unstable because the target itself changes at every iteration
        self.target_net = DQN(input_shape, n_actions)

        # make the target_network as same as policy network (sometimes)
        # initially, the target network and policy network have identical weights.
        # However, periodic update is done for stable training.
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Because it is not used for learning or updating its parameters directly during training. 
        # Instead, target_net is used to provide stable Q-value targets for training the policy network
        # .eval() turns off batchnorm / dropout, which behave differently during training and evaluation.
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = 1e-3)
        self.n_actions = n_actions


    ## Let's select our action!!
    ## We use Epsilon-Greedy Policy, which balances between "Exploration" and "Exploitation".
    ## We define certain threshold of epsilon. 
    ## For example, if epsilon = 0.2,
    ## We have 0.8 chance of Exploitation and 0.2 chance of Exploration.
    ### The use of random.random() introduces randomness into the decision-making process, 
    # which is crucial in the early stages of training to help the agent learn effectively by exploring different actions.
    def select_action(self, state, epsilon_threshold):

        
        # We need exploitation
        if random.random() > epsilon_threshold:

            ## Selecting action only requires Q-values, no gradients are required, so turn off
            with torch.no_grad():
                state_np = np.array(state)
                state_tensor = torch.tensor([state_np], dtype=torch.float32).to(next(self.parameters()).device)
                # Dynamic Retrival of device where params live 
                q_values = self.policy_net(state_tensor)

                # How would the Q value shape look like?
                ## Q_value is attained through policy net
                ## If n actions go into policy network, it outputs shape of (n, n_actions)
                ## so in the n batch, (n, n x q_values)
                # tensor([[1.2, 2.5],    # Q-values for first state in the batch
                        # [0.8, 3.1],    # Q-values for second state in the batch
                        # [1.0, 1.8]])   # Q-values for third state in the batch

                # The numbers in that tensor are: estimates of the expected cumulative reward
                # 2 actions (row) , of 3 states    
                # 1.2 and 2.5 is the final reward of State 1 when action 1 and action 2 is done separately
                # 0.8 and 3.1 is the final reward of State 2 when action 1 and action 2 is done separately
                # 1.0 and 1.8 is the final reward of State 3 when action 1 and action 2 is done separately

                # We want the highest q value.
                # q_values.max(1) scans through the 1st dim of q_value
                # In this case, max[1.2, 2.5] ,  max[0.8, 3.1], max[1.0, 1.8]

                # q_values.max(1) returns a tuple (values, indicies)
                # values = tensor([2.5, 3.1, 1.8])   # Maximum Q-values for each state in the batch
                # indices = tensor([1, 1, 1])        # Indices of actions that have the max Q-values

                # Then, [1] retrieves indicies.
                # In this case, indicies of max values.
                # .item() convert the tensor to a Python scalar which is needed because actions are represented as integers
                # We retrieve index of action list for choosing a specific action.
                return q_values.max(1)[1].item()

        # We need exploration
        else:
            return random.randrange(0, self.n_actions)
            # This returns a random integer between 0 and n_actions - 1
            # This random integer is an index for action.
            # So random index of action is selelcted

    def optimize_model(self, replay_buffer, batch_size, gamma):

        # Edge case:
            # If we have less experience than the batch_size, we do nothing
            # We need enough experience to sample

        if len(replay_buffer) < batch_size:
            return 

        state_batch, action_batch, reward_batch, next_state_batch, end_batch = replay_buffer.sample(batch_size)
        device = next(self.parameters()).device

        # Why do we code like this:
        ## In CartPole-v1, we have 4 features that describe 1 state (cart position, cart velocity, pole angle, pole angular velocity)
        ## so 1 state is (x, x', theta, theta')

        # Suppose we have batch of 32:
        # state_batch = (32, 4) 
        ### [[ 0.05, -0.02, 0.03, 0.07], --s1
           # [-0.01, 0.01, -0.05, 0.02], --s1  (Can have multiple s1s)
            # ...                  
           # [ 0.06, -0.03, 0.04, 0.01]]

        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(device)

        # action_batch = (32, )
        ## After unsqueezing, we will have (32, 1)
        ## gives what actions were done in each states
        ## dtype must be int64, because we have discrete action indexes
        # action_batch = (32, 1)
        ## [[0.0],
         ## [1.0],
         ## [0.0],
         ## [0.0],
         ## [1.0],
         ## ...]

        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(device)

        # reward_batch
        # would be (32, 1). Same as action_batch, (32,) -> unsqueeze and -> (32, 1)

        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)

        # We need a next state batch.
        # next state is also the same with the state batch, so:
        # next_state_batch = (32, 4)

        next_state_batch = torch.tensor(next_state_batch, dtype =torch.float32).to(device)

        # We need a end batch, to indicate whether the episode has ended or not.
        # A batch of flags indicating whether the episode ended after taking the action.
        # gamma * next_q_values * (1 - done_batch) becomes 0.
        # In this case, only the immediate reward (reward_batch) contributes to expected_q_values. 
        # done_batch is represented with dtype torch.float32 instead of torch.int64 
        # because of its use in mathematical operations involving floating-point arithmetic
        # Specifically, expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
        
        end_batch = torch.tensor(end_batch, dtype=torch.int64).unsqueeze(1).to(device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        # after self.policy_net, the output is shaped (batch_size, number_of_actions), 
        # where each row contains q_values for all possible actions
        # Example of the policy_net output: 
        # (batch_size, n_actions) = (32, 2)  (total of 2 actions)

        # [[1.2, 2.5],    # Q-values for actions 0 and 1 in state 1
        #  [0.8, 3.1],    # Q-values for actions 0 and 1 in state 2
        #   ...
        #  [1.0, 1.8]]    # Q-values for actions 0 and 1 in state 32   # Can have multiple same states (s1, s2, s1, s3, ... 32 of them)
       
        # Now we have the q_values, we want the q_values of our "actually taken" actions.
        # The q_values are stored in the n_actions dimension, which is the 1st dimension (python starts with 0)
        # and we gather the ones that were actually our action. In this example, it would be one between 1.2 and 2.5, 0.8 and 3.1, so on)
        
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        # after self.target_net, the output is shaped (batch_size, number_of_actions)
        # .max(1) returns a tuple, where:
            # The first element ([0]) contains the maximum Q-values for each state.
            # The second element ([1]) contains the indices of the actions that yielded these maximum Q-values.
        # Since we only want the maximum values, we write [0]
        # It would return something like: [2.2, 3.0, ..., 1.7]  # Maximum Q-values for each next state (shape: (32,))
        # .detach() removes the next_q_values tensor from the computation graph, 
        # which means that gradients are not calculated for this part of the network during backpropagation.
        # This is because we don't want to update target network while calculating the target q values.
        # Since we have (32,), we have to add an extra dim through .unsqueeze(1)

        expected_q_values = reward_batch + (gamma * next_q_values * (1 - end_batch))
        # Here we take in the case of episode end, by multiplying (1 - end_batch) 
        # if this is 0, I mean, end_batch == 1, we have no next_q_values (multiply by 0)

        # COmputing loss between target q values and current policy q values
        loss = F.mse_loss(q_values, expected_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Optional: Gradient clipping
        self.optimizer.step()

        # After calling backward(), the gradients are acculmulated, so we zero_grad it
        # optimizer object (self.optimizer) that performs a single optimization step
        # to update the model's parameters based on the calculated gradients.

    ## Update Target Network
    def update_target_network(self):
        """
        Updates the target network to match the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())





