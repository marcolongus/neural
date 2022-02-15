import matplotlib.pyplot as plt
import numpy as np

model_name = 'model-1644207649'

def create_acc_loss_graph(madel_name):
	contents = open("model.log","r").read().split('\n')

	times = []
	accuracies = []
	losses = []
	
	val_accs = []
	val_losses = []

	epoch_ticks = []
	end_of_epoch = 0
	for c in contents:
		if model_name in c:
			name, epoch, timestamp, acc, loss, val_acc, val_loss = c.split(",")

			if end_of_epoch != int(epoch):
				end_of_epoch += 1
				epoch_ticks.append(float(timestamp))
					
			times.append(float(timestamp))
			accuracies.append(float(acc))
			losses.append(float(loss))

			val_accs.append(float(val_acc))
			val_losses.append(float(val_loss))

	fig = plt.figure()
	
	ax1 = plt.subplot2grid((2,1), (0,0))
	ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
	ax1.set_ylim(0.4,1)
	ax2.set_ylim(0, 0.4)
	
	ax1.set_xticks(epoch_ticks[::5])
	ax2.set_xticks(epoch_ticks[::5])
	
	ax1.plot(times, accuracies, label='Train acc')
	ax1.plot(times, val_accs, label='Test acc')
	ax1.legend()
	
	ax2.plot(times, losses, label='Train loss')
	ax2.plot(times, val_losses, label='Test loss')
	ax2.legend()
	
	plt.show()

create_acc_loss_graph(model_name)