################### Loss Function #######################

# Plot loss as a function of epochs
import matplotlib.pyplot as plt
epochs = np.arange(100)

plt.plot(epochs, Loss_train, label = 'Loss_train')
plt.plot(epochs, Loss_valid, label= 'Loss_Valid')
plt.plot(epochs, Loss_test, label = 'Loss_Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
