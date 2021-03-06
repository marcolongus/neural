import torch.optim as optim
from data import *
from convolution import *

# Load data
training_data = np.load("training_data.npy", allow_pickle=True)

image = (
    torch.tensor(np.array([image[0] for image in training_data])).view(-1, 50, 50)
    / 255.0
)

target = torch.tensor(
    np.array([image[1] for image in training_data]), dtype=torch.float32
)

# Init CNN
net = Convolution()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


# Validation test size 10 %
VAL_PCT = 0.1
val_size = int(len(image) * VAL_PCT)


def train(net, EPOCHS=1, BATCH_SIZE=100):
    train_image = image[:-val_size]
    train_target = target[:-val_size]
    for epoch in range(EPOCHS):
        print(f"EPOCH:{epoch}")
        for i in range(0, len(train_image), BATCH_SIZE):
            batch_image = train_image[i : i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_target = train_target[i : i + BATCH_SIZE]
            net.zero_grad()
            outputs = net(batch_image)
            loss = loss_function(outputs, batch_target)
            loss.backward()
            optimizer.step()
        print(f"EPOCH{epoch}, LOSS:{loss}")


def test(net):
    test_image = image[-val_size:]
    test_target = target[-val_size:]
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_image)):
            real_class = torch.argmax(test_target[i])
            net_out = net(test_image[i].view(-1, 1, 50, 50))
            predicted_class = torch.argmax(net_out)
            if i < 10:
                ...
                # plt.title(f"Class: {real_class}, Prediction {predicted_class}")
                # plt.imshow(test_image[i])
                # plt.show()
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct / total, 3))


def main():
    train(net, EPOCHS=2, BATCH_SIZE=100)
    test(net)


if __name__ == "__main__":
    main()
