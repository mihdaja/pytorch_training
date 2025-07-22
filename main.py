import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim.sgd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# NEURAL NETWORK definition 

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs): # num inputs reprezents the number of inputs, for example 28x28 in the case of the mnist dataset i'll use
        super().__init__() # this allows me to use the methods in nn.Module, which are very useful for making nn's

        self.layers = nn.Sequential( # initializes a sequential container, which will be important for using the .forward method later on to move the input data through the network

            # first hidden layer
            nn.Linear(num_inputs, 30), # takes in the number of inputs orderd in a matrix: A(1,num_inputs) then passes out a matrix: A1(1, 30)

            nn.ReLU(), # passes the matrix in only positives (0 if x < 0 and x if x >= 0), easier to differentiate than a sigmoid

            #second hidden layer
            nn.Linear(30, 20),
            nn.ReLU(),
            
            # output layer
            nn.Linear(20, num_outputs), 
        )

    def forward(self, x): # passes the input through all the neural layers and outputs the resulting logits through the return function
        logits = self.layers(x)
        return logits
        

# DATASETS & DATALOADERS for training and testing

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])


X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y): #we initialize the dataset using an array of input data, "X", and an array of output answers, "y"
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x,one_y # returns the tensor and answer on position "index" in the arrays we passed to this dataset
    
    def __len__(self): #might be used by the DataLoader in order to know how long to iterate over the dataset
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)

test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0, # how many workers will prepare the data in the background, while the program trains the nn. If set to zero, the program will have to wait to load the data for every epoch. 
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

#TRAINING MODEL

torch.manual_seed(123)

model = NeuralNetwork(num_inputs=2, num_outputs=2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5) # Stochastic Gradient Descent step of the function

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels) # calculates the error of the model

        optimizer.zero_grad() # this sets the grad of all the parameters in the function to "None", in order to prevent the accumulation of gradients, leading to inneficient or wrong steps
        loss.backward() # calculates the gradient of all the parameters in the model using the chain rule
        optimizer.step() # applies the gradient using the formula P_t = P_(t-1) - y * g_t, where P_(t-1) is the parameter from the last epoch, g_t is the gradient calculated with the backward function for this param and y is the learning rate defined in the declaration


        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")
    



def compute_accuracy(model, dataloader):

    model = model.eval()

    nr_correct = 0.0
    total = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim = 1)

        correct = predictions == labels
        nr_correct += torch.sum(correct)
        total += len(correct)
    return (nr_correct/total).item()

# print(compute_accuracy(model, train_loader))

torch.save(model.state_dict(), "model.pth") # save the model's weights and biases to the model.pth file
model2 = NeuralNetwork(2, 2)
model2 .load_state_dict(torch.load("model.pth", weights_only=True)) # load the model 

print(compute_accuracy(model2, train_loader)) # checking the new model we just copied