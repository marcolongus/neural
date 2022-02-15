import torch.optim as optim
import time
from data import *
from convolution import *

# Load datas
training_data = np.load("training_data.npy", allow_pickle=True)

image = (
    torch.tensor([image[0] for image in training_data], dtype=torch.float32, device=device).view(-1, 50, 50)
    / 255.0
)
target = torch.tensor(
    [image[1] for image in training_data], dtype=torch.float32, device=device
)

# Init CNN, optimizer and loss function
net = Convolution().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Validation test size 10 % of data
VAL_PCT = 0.1
val_size = int(len(image) * VAL_PCT)
print(f"Validation test size: {val_size}")
print(f"Train test size: {len(image) - val_size}")

BATCH_SIZE = 200
EPOCHS = 30

MODEL_NAME = f"model-{int(time.time())}"

def fwd_pass(image, target, train=False):
    if train:
        net.zero_grad()
    outputs = net(image)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, target)]
    try:
        accuracy = matches.count(True) / (len(matches) )
    except:
        print('Exception')
        accuracy = 0
    loss = loss_function(outputs, target)

    if train:
        loss.backward()
        optimizer.step()
    return accuracy, loss


def test_model(size):

    test_image = image[-val_size:]
    test_target = target[-val_size:]

    random_start = np.random.randint(len(test_image) - size)

    image_tm = test_image[random_start : random_start + size]
    target_tm = test_target[random_start : random_start + size]

    with torch.no_grad():
        val_acc, val_loss = fwd_pass(image_tm.view(-1, 1, 50, 50), target_tm)
    
    return val_acc, val_loss


def train_model():
    with open('model.log', 'w', encoding = 'utf-8') as f:
        train_image = image[:-val_size]
        train_target = target[:-val_size]
        
        for epoch in range(EPOCHS):
            print(f'EPOCH:{epoch}')
            for i in range(0, len(train_image), BATCH_SIZE):
                batch_image = train_image[i : i + BATCH_SIZE].view(-1, 1, 50, 50)
                batch_target = train_target[i : i + BATCH_SIZE]

                acc, loss = fwd_pass(batch_image, batch_target, train=True)
                if i % 100 == 0:
                    val_acc, val_loss = test_model(size=100)
                    f.write(
                        f"{MODEL_NAME}, {epoch}, {round(time.time(),3)}, {round(float(acc),2)}, {round(float(loss),4)}, {round(float(val_acc),4)}, {round(float(val_loss),4)}\n"
                    )
                



train_model()
