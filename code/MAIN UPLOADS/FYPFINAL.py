import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torchvision.models import resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
import time
import matplotlib.pyplot as plt
import numpy as np

def setup_unique_output_file(base_dir, base_name):
    count = 1 # start count at 1
    while True: # loop until a unique filename is found
        print_path = os.path.join(base_dir, f"{base_name}{count}.txt")
        if not os.path.exists(print_path): # new file name found, break the loop
            break
        count += 1
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler(print_path, 'w', 'utf-8'),
                                  logging.StreamHandler()]) # log to file and console

    return print_path

print_path = setup_unique_output_file('C://Users//zeesh//uni notes//Final Year Project//scripts//testoutputs', 'testprint')
# both models stored here for comparison
resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet34_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

class FaceClassifier(nn.Module):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        self.features = resnet18_model
        self.features.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.features(x)
        return x
# compare resnet18 and resnet34    


class CustomDataset(Dataset):
    def __init__(self, data_folder, labels_csv, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)

        self.file_labels = {row['File']: row['Label'] for _, row in self.labels_df.iterrows()}

        self.file_names = [file for file in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, file))]

        # define mapping for labels
        self.label_mapping = {
            'light-male': 0,
            'dark-male': 1,
            'light-female': 2,
            'dark-female': 3,
            'non-face': 4
        }

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.file_names[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.label_mapping[self.file_labels[self.file_names[idx]]]

        return image, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FaceClassifier().to(device)

if isinstance(model.features, type(resnet18_model)):
    logging.info("Model Architecture: ResNet18")
elif isinstance(model.features, type(resnet34_model)):
    logging.info("Model Architecture: ResNet34")

v2_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

faces_training_dataset = CustomDataset(
    data_folder='C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//all5classes-distorted',
    labels_csv='C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//distorted_labels.csv',
    transform=v2_transform
)
binary_training_dataset = CustomDataset(
    data_folder='C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//binary-class',
    labels_csv='C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//binarylabels.csv',
    transform=v2_transform
)

def convert_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    time_str = ""
    if hours > 0:
        time_str += f"{hours}h "
    if minutes > 0:
        time_str += f"{minutes}m "
    if seconds > 0:
        time_str += f"{seconds}s"

    return time_str.strip() # function to format time

dataset_size = len(faces_training_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(faces_training_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test 64 and 128 batch size
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False)

criterion = nn.CrossEntropyLoss()

optimizer01 = optim.Adam(model.parameters(), lr=0.01)
optimizer001 = optim.Adam(model.parameters(), lr=0.001)
optimizer0001 = optim.Adam(model.parameters(), lr=0.0001)

sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optimizer0001

if isinstance(optimizer, optim.Adam):
    logging.info("Optimizer: Adam")
elif isinstance(optimizer, optim.SGD):
    logging.info("Optimizer: SGD")
# test 0.001 and 0.0001 learning rate and adam and sgd optimizer
logging.info(f"Training Dataset size: {train_size}" )
logging.info(f"Training batch size: {train_dataloader.batch_size}")
logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print("Training started...")
epochs = 25
# test 10 and 25 epochs
torch.cuda.empty_cache()

time_so_far = 0
start_total_time = time.time()  # timer for the entire training process
for epoch in range(epochs):
    start_epoch_time = time.time()  # timer for the current epoch
    # set the model to training mode
    model.train() 
    
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # reset the model parameters gradient to zero
        outputs = model(inputs) # pass the training data to the model for the forward pass
        loss = criterion(outputs, labels) # calculate the loss between the model predictions and the actual labels
        loss.backward() # this is the backward pass where the model calculates the gradients of the loss with respect to the model parameters
        optimizer.step() # this is the optimization step where the model updates the parameters based on the gradients

    end_epoch_time = time.time()  # end timer for the current epoch
    epoch_duration = end_epoch_time - start_epoch_time
    time_so_far += epoch_duration
    logging.info(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Duration: {convert_time(epoch_duration)} seconds')
    print(f'Predicted time left = {convert_time(time_so_far / (epoch + 1) * (epochs - epoch - 1))}')

# total duration timer
end_total_time = time.time()
alt_end_total_time = time_so_far
total_duration = end_total_time - start_total_time
logging.info(f"Average time per epoch:\n {convert_time(time_so_far / epochs)}")
logging.info(f"Training completed. Total duration:\n {convert_time(total_duration)}")

torch.cuda.empty_cache()

model.eval() 
all_preds = []
all_labels = []
all_scores = []  

losses = []
accuracies = []

with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
        losses.append(val_loss.item())

        _, preds = torch.max(outputs, 1)
        scores = torch.softmax(outputs, dim=1) 
        all_scores.extend(scores.cpu().numpy())
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    accuracies.append(accuracy)
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    class_names = ['light-male', 'dark-male', 'light-female', 'dark-female', 'non-face']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    all_scores = np.array(all_scores)

    for i, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(all_labels, all_scores[:, i], pos_label=i)
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])


cm = confusion_matrix(all_labels, all_preds)
# calculate average precision, recall, and f1 across all classes
avg_accuracy = accuracy_score(all_labels, all_preds)
avg_precision = precision_score(all_labels, all_preds, average='macro')
avg_recall = recall_score(all_labels, all_preds, average='macro')
avg_f1 = f1_score(all_labels, all_preds, average='macro')

# print average performance metrics for model
logging.info('Average Performance Metrics:')
logging.info(f'  Average Accuracy: {avg_accuracy:.4f}')
logging.info(f'  Average Precision: {avg_precision:.4f}')
logging.info(f'  Average Recall: {avg_recall:.4f}')
logging.info(f'  Average F1 Score: {avg_f1:.4f}')
logging.info(f'  Average of Accuracy and F1 Score: {(avg_accuracy + avg_f1) / 2:.4f}')
s
accuracy_list = [accuracy] * len(class_names)

# print performance metrics for each class
logging.info('Performance Metrics by Class:')
for i, (class_name, prec, rec, f1_score) in enumerate(zip(class_names, precision, recall, f1)):
    logging.info(f'Class: {class_name}')
    TP = cm[i, i]
    FP = cm[:, i].sum() - cm[i, i]
    FN = cm[i, :].sum() - cm[i, i]
    TN = cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
    manual_accuracy = (TP + TN) / (TP + TN + FP + FN)
    logging.info(f'  Accuracy: {manual_accuracy.round(4)}')
    logging.info(f'  Precision: {prec.round(4)}')
    logging.info(f'  Recall: {rec.round(4)}')
    logging.info(f'  F1 Score: {f1_score.round(4)}')
    logging.info(f'  Error Rate: {(1 - manual_accuracy) * 100:.2f}%')
    logging.info(f'  Confusion Matrix for {class_name}:')
    logging.info(f'             TRUE   FALSE')
    logging.info(f'            --------------')
    logging.info(f' POSITIVE    | {TP} | {FP} |')
    logging.info(f'            --------------')
    logging.info(f' NEGATIVE    | {TN} | {FN} |')
    logging.info(f'            --------------')
    logging.info(f"\nClass '{class_name}' misclassification analysis:")
    total_misclassified = cm[i, :].sum() - cm[i, i]
    logging.info(f"Total misclassified: {total_misclassified}")
    for j, other_class in enumerate(class_names):
        if i != j:
            logging.info(f"  Misclassified as '{other_class}': {cm[i, j]}") # print each time class i is misclassified as class j


fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# loss curve
axs[0, 0].plot(losses, label='Validation Loss')
axs[0, 0].set_title('Loss Over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_xlim([0, epochs])  # set x-axis limits
axs[0, 0].legend()

# extract weights from the model for histogram
weights = []
for param in model.parameters():
    if param.requires_grad:
        weights += list(param.detach().cpu().numpy().flatten())

# histogram of weights
axs[0, 1].hist(weights, bins=30, label='Model Weights')
axs[0, 1].set_title('Histogram of Model Weights')
axs[0, 1].set_xlabel('Weight')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].legend()

# ROC curve for each class
for i, class_name in enumerate(class_names):
    axs[1, i % 2].plot(fpr[class_name], tpr[class_name], label=f'{class_name} (area = {roc_auc[class_name]:.2f})')
    axs[1, i % 2].plot([0, 1], [0, 1], 'k--')
    axs[1, i % 2].set_xlim([0.0, 1.0])
    axs[1, i % 2].set_ylim([0.0, 1.05])
    axs[1, i % 2].set_xlabel('False Positive Rate')
    axs[1, i % 2].set_ylabel('True Positive Rate')
    axs[1, i % 2].set_title('ROC Curve')
    axs[1, i % 2].legend(loc="lower right")

plt.tight_layout()
plt.show()


print("Training and validation completed.")

# Save the model
#torch.save(model.state_dict(), 'C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//face_classifier.pth')