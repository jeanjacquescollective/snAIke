import torch # import torch which is a machine learning library
import torch.nn as nn # import neural network module from torch
import torch.nn.functional as F # import functional module from torch
import torch.optim as optim  # import optimization module from torch
import os # import os module

class Linear_QNet(nn.Module): # create a class called Linear_QNet which inherits from nn.Module
    def __init__(self, input_size, hidden_size, output_size): # define the constructor for Linear_QNet
        super().__init__() # call the constructor of nn.Module
        self.linear1 = nn.Linear(input_size, hidden_size) # define the first linear layer
        self.linear2 = nn.Linear(hidden_size, output_size) # define the second linear layer

    def forward(self, x): # define the forward function, x is the tensor that is passed through the network
        x = F.relu(self.linear1(x)) # pass the input through the first linear layer and apply the activation function
        x = self.linear2(x) # pass the output of the first linear layer through the second linear layer, we dont need to apply the activation function here
        return x # return the output of the second linear layer

    def save(self, file_name='model.pth'): # define the save function
        model_folder_path = './model' # define the model folder path
        if not os.path.exists(model_folder_path): # if the model folder does not exist
            os.makedirs(model_folder_path) # make the model folder
        file_name = os.path.join(model_folder_path, file_name) # create the file path
        torch.save(self.state_dict(), file_name) # save the model

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # choose the optimizer
        self.criterion = nn.MSELoss() # choose the loss function, mean squared error loss
    
    def train_step(self, state, action, reward, next_state, done): # define the train step function
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        # (n, x) -> (n, 1)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x) -> (n, 1)
        if len(state.shape) == 1:
            # (1, x) -> (1, 1)
            # unsqueeze adds a dimension to the tensor
            # unsqueeze(0) adds a dimension at the specified index
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action).item()] = Q_new

        # max is only one parameter
        # 2: Q_new = r + y * max(next_predicted Q Value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        # empty gradient
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
