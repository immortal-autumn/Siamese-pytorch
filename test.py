import json
from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from siamese import Siamese

test_set = Path('./testdata')
model_path = Path('./logs')

tests = []
for animal_name in test_set.glob('*'):
    # print(f'- {animal_name.name}')
    for img_path in animal_name.glob('*'):
        tests.append((img_path.absolute(), animal_name.name))

# TODO: Make the testset balanced
num_of_true = 0
atest = []
# Construct test sets and ground truth
for i in range(len(tests)):
    for j in range(i + 1, len(tests)):
        path1, animal1 = tests[i]
        path2, animal2 = tests[j]
        if animal2 == animal1:
            num_of_true += 1
        atest.append((path1, path2, animal1 == animal2))  # Ground Truth


# print(num_of_true)

def evaluation(predict, truth):
    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(truth, predict)
    disp.figure_.suptitle("Confusion Matrix for {}".format(i))
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()
    # Other evaluations
    report = classification_report(truth, predict, digits=5, output_dict=True)
    print(
        f"Classification report for classifier:\n"
        f"{classification_report(truth, predict, digits=5)}\n"
    )
    return report


def main(model_pth):
    global num_of_true
    num_of_true_dp = num_of_true
    model = Siamese(model_path=model_pth)
    Y = []
    prediction = []
    process = 0
    total = len(atest)
    for tup in atest:
        # if process == 1000:
        #     break
        p1, p2, res = tup
        if not res:
            if num_of_true_dp != 0:
                num_of_true_dp = num_of_true_dp - 1
            else:
                continue
        img_1 = Image.open(p1)
        img_2 = Image.open(p2)
        Y.append(res)
        result = model.detect_image(img_1, img_2)
        prediction.append(True if result > 0.5 else False)
        process += 1
        print(f"Processing...({process}/{num_of_true * 2})")
    # print(Y)
    # print(prediction)
    return evaluation(prediction, Y)


evaluation_feedback = {}
for i in model_path.glob('*.pth'):
    print(f'loading model {i}...')
    evaluation_feedback[i.name] = main(i)

print(evaluation_feedback)
with open('output.json', 'w+') as f:
    json.dump(evaluation_feedback, f, indent=4)
