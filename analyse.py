import json

import matplotlib.pyplot as plt

with open('output.json', 'r') as f:
    result = json.load(f)

# plotting = {}
indexs = []
losses = []
val_losses = []
accuracies = []

for k, v in result.items():
    name = k.replace(".pth", "").split('-')
    index = int(name[0][2:])
    loss = float(name[1].replace("loss", ""))
    val_loss = float(name[2].replace("val_loss", ""))
    accuracy = v["accuracy"]
    # Add to list
    indexs.append(index)
    losses.append(loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)
    # plotting[index] = {
    #     "loss": loss,
    #     "val_loss": val_loss,
    #     "accuracy":  accuracy,
    # }

    print(index, loss, val_loss, accuracy)

print(f"Max Loss: {max(losses)}\nMin Loss: {min(losses)}")
print(f"Max Validation Loss: {max(val_losses)}\nMin Validation Loss: {min(val_losses)}")
print(f"Max Accuracy: {max(accuracies)} - {accuracies.index(max(accuracies))}"
      f"\nMin Accuracy: {min(accuracies)} - {accuracies.index(min(accuracies))}")

plt.xlabel('Index')
plt.ylabel('Value')
# for k, v in plotting.items():
#     plt.plot(k, v['loss'])
plt.plot(indexs, losses, label="loss")
plt.plot(indexs, val_losses, label="validation_loss")
plt.plot(indexs, accuracies, label="accuracy")
# print((k, v['loss']) for k, v in plotting.items())
plt.legend()
plt.show()
