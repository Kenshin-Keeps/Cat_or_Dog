'''
1. Clone Yolov5 github repository 
Use command -
git clone https://github.com/ultralytics/yolov5.git

2. Goto to cloned repository
cd yolov5

3. Install all the required packages
pip install -r requirements.txt

'''
# Import required libraries
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import os
from glob import glob
import yaml

# Copy images to appropriate folder
imgs_dir = '/dataset/export/images'
os.makedirs(imgs_dir, exist_ok=True)
copy_tree("/dataset/images", imgs_dir)

# Copy annotations to appropriate folder
text_labels_dir = '/dataset/export/labels'
os.makedirs(text_labels_dir, exist_ok=True)
copy_tree("/dataset/annotations", text_labels_dir)

# Split data in train validation and test
img_list = glob(imgs_dir + '/*')
train_img, valid_img = train_test_split(
    img_list, test_size=0.1, random_state=42)
train_img, test_img = train_test_split(
    train_img, test_size=0.2, random_state=42)

# Write images to corresponding text files
with open('/dataset/train.txt', 'w') as f:
    f.write('\n'.join(train_img) + '\n')
with open('/dataset/val.txt', 'w') as f:
    f.write('\n'.join(valid_img) + '\n')
with open('/dataset/test.txt', 'w') as f:
    f.write('\n'.join(test_img) + '\n')

# Set correct configuration in data file
with open('/dataset/data.yaml', 'w') as f:
    data = {
        'train': '/dataset/train.txt',
        'val': '/dataset/val.txt',
        'test': '/dataset/test.txt',
        'nc': 2,
        'names': ['cat', 'dog']
    }
    yaml.dump(data, f)

# specify "yolov5s" during training and set batch 8
'''
4. Run the Command for Training the model using pre-trained weight
python train.py \
--img 416 \
--batch 16 \
--epochs 4000 \
--data /dataset/data.yaml \
--cfg ./models/yolov5s.yaml \
--weights yolov5s.pt \
--name cat_and_dog_yolov5s_results

5. Goto cloned repository
cd yolov5

6. Run the Command for Testing the model using best weight
python val.py \
--weights /runs/train/cat_and_dog_yolov5s_results/weights/best.pt \
--data /dataset/data.yaml \
--img 416 \
--batch 16 \
--conf 0.5 \
--task test
'''
