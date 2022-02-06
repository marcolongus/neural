from training import *
import time

MODEL_NAME = f"model-{int(time.time())}"


def fwd_pass(image=image, target=target, train=False):
    if train:
        net.zero_grad()
    outputs = net(image)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, target)]
    try:
        accuracy = matches.count(True) / (len(matches) )
    except:
        accuracy = 0
    loss = loss_function(outputs, target)

    if train:
        loss.backward()
        optimizer.step()
    return accuracy, loss


def test_model(size):

    test_image = image[-val_size:]
    test_target = target[-val_size:]

    random_start = np.random.randint(len(image) - size)

    image_tm = test_image[random_start : random_start + size]
    target_tm = test_target[random_start : random_start + size]

    with torch.no_grad():
        val_acc, val_loss = fwd_pass(image_tm.view(-1, 1, 50, 50), target_tm)
    return val_acc, val_loss


def train_model():
    train_image = image[:-val_size]
    train_target = target[:-val_size]

    for epoch in range(EPOCHS):
        for i in range(0, len(train_image), BATCH_SIZE):
            batch_image = train_image[i : i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_target = train_target[i : i + BATCH_SIZE]

            acc, loss = fwd_pass(batch_image, batch_target, train=True)
            if i % 50 == 0:
                val_acc, val_loss = test_model(size=100)
                print(
                    f"{MODEL_NAME}, {round(time.time(),3)}, {round(float(acc),2)}, {round(float(loss),4)}, {round(float(val_acc),2)}, {round(float(val_loss),4)}"
                )

train_model()
