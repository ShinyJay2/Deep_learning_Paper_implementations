import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, number_of_actions):
        super().__init__()


        # input_shape is the shape of the state that will be used as input to the model.
        # number_of_actions is the output dimension of the NN.
        # So we map state (features) -> q_values
        # the q_value size is shaped as the number of possible actions.
        ## So the DQN ouput:
            # output_size = number_of_actions
            # output content = q_values corresponding to each action

        # The CartPole-v1 environment, environment's state is a 1-dimensional NumPy array (vector) containing 
        # four continuous numerical values.
            # 1. Cart Position: Position of the cart along the track.
            # 2. Cart Velocity: Speed of the cart.
            # 3. Pole Angle: Angle of the pole with respect to the vertical.
            # 4. Pole Velocity at Tip: Rate of change of the pole's angle.

        ## state shpae = (4,)
        # We need only integer to pass in, not tensor nor vector.
        # So we use input_shape[0] to only fetch 4. 

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_actions)
        )

    def forward(self, x):
        return self.fc(x)
