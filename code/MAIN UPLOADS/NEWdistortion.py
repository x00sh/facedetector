import os
from PIL import Image
import shutil
import torchvision.transforms as transforms
import torchvision.transforms.v2 as T
import torch
import csv

base_directory = 'C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//'

distorted_subdirectories = ['light-male-distorted', 'dark-male-distorted', 'light-female-distorted', 'dark-female-distorted', 'non-face-distorted', 'all5classes-distorted']

def deletion(base_directory, distorted_subdirectories):
    # check over each subdirectory
    for subdir in distorted_subdirectories:
        # create the path to the subdirectory
        subdirectory_path = os.path.join(base_directory, subdir)
        
        # check if the subdirectory exists
        if os.path.exists(subdirectory_path):
            # delete the whole subdirectory
            shutil.rmtree(subdirectory_path)
            print(f"Subdirectory {subdir} erased successfully.")
        else:
            print(f"Subdirectory {subdir} does not exist.")

# erase the current data to make room for the new distorted data
deletion(base_directory, distorted_subdirectories)

subdirectories = ['light-male', 'dark-male', 'light-female', 'dark-female', 'non-face']

# define the possible augemtnations
augmentation_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomErasing(p=0.5, scale=(0.01, 0.165), ratio=(0.3, 3.3), value=0, inplace=False),  # Halved max size
    T.GaussianBlur(kernel_size=5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.RandomPerspective(distortion_scale=0.5, p=0.5), # test out new perspective distortion
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
    T.ElasticTransform(alpha=50.0), # test out new elastic distortion and remember to mention in report
    T.Resize((224, 224)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])

to_pil = transforms.ToPILImage()

def augment_and_save_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            # load and convert image
            image = Image.open(input_path).convert('RGB')
            for i in range(3):
                output_path = os.path.join(output_folder, f'augmented_{i+1}_{filename}')
                transformed_tensor = augmentation_transform(image)
                transformed_image = to_pil(transformed_tensor)
                transformed_image.save(output_path)
            # copy the original image
            shutil.copy(input_path, output_folder)
    print(f"{os.path.basename(input_folder)} has been completed")

print("Starting image augmentation...")
for subdir in subdirectories:
    print(f"Processing {subdir}...")
    input_path = os.path.join(base_directory, subdir)
    output_path = os.path.join(base_directory, f"{subdir}-distorted")
    augment_and_save_images(input_path, output_path)

# consolidate all images
consolidated_path = os.path.join(base_directory, "all5classes-distorted")
os.makedirs(consolidated_path, exist_ok=True)
for subdir in subdirectories:
    distorted_path = os.path.join(base_directory, f"{subdir}-distorted")
    for filename in os.listdir(distorted_path):
        shutil.copy(os.path.join(distorted_path, filename), consolidated_path)

print("All subdirectories have been processed and images consolidated in 'all5classes-distorted'.")

labels = [
    'light-male',
    'dark-male',
    'light-female',
    'dark-female',
    'non-face'
]

base_path = 'C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//'

# create full folder paths for all distorted subdirectories
all_paths = [os.path.join(base_path, folder) for folder in distorted_subdirectories]

# define the path to the CSV file
output_csv_path = 'C://Users//zeesh//Downloads//ALLFYPDATASETS//4CLASSSPLITDATA//distorted_labels.csv'

# collect the data for CSV
data = []
for label, folder_path in zip(labels, all_paths):
    if not os.path.exists(folder_path):
        print(f"Warning: The folder {folder_path} does not exist.")
        continue
    file_names = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    label_data = [{'File': file, 'Label': label} for file in file_names]
    data.extend(label_data)

# write to CSV
with open(output_csv_path, 'w', newline='') as csv_file:
    fieldnames = ['File', 'Label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f'CSV file "{output_csv_path}" created successfully with distorted folder paths.')


