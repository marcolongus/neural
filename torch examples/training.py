import torch.optim as optim
from tqdm import trange
from data import *
from convolution import *

# Load data
training_data = np.load("training_data.npy", allow_pickle=True)

# Init CNN
net = Convolution()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

image = (
    torch.tensor(np.array([image[0] for image in training_data])).view(-1, 50, 50)
    / 255.0
)
target = torch.tensor(np.array([image[1] for image in training_data]), dtype=torch.float32)


# Validation test size 10 % of data
VAL_PCT = 0.1
val_size = int(len(image) * VAL_PCT)

print("Validation test size:", val_size)

train_image = image[:-val_size]
train_target = target[:-val_size]

test_image = image[-val_size:]
test_target = target[-val_size:]


print(test_image.shape)
print(train_image.shape)

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_image), BATCH_SIZE)):
        batch_image = train_image[i : i + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_target = train_target[i : i + BATCH_SIZE]

        # Zero gradient
        net.zero_grad()
        outputs = net(batch_image)
        loss = loss_function(outputs, batch_target)
        loss.backward()
        optimizer.step()


print("LOSS:", loss)