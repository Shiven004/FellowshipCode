import os
import shutil

root_path = 'CUB_200_2011'
imgs_path = 'CUB_200_2011/images'
bb_path = 'CUB_200_2011/bounding_boxes.txt'
img_list_path = 'CUB_200_2011/images.txt'
classes_path = 'CUB_200_2011/classes.txt'
split_path = 'CUB_200_2011/train_test_split.txt'

def make_dir(dir_path):
	if not os.path.isdir(dir_path):
		print('Creating dir: {}'.format(dir_path))
		os.mkdir(dir_path)

classes = []
# Read classes
with open(classes_path, 'r') as file:
	for line in file.readlines():
		classes.append(line.split('\n')[0].split(' ')[1])
#print(classes)

# Create directories
make_dir(os.path.join(root_path, 'train'))
make_dir(os.path.join(root_path,'val'))
for c in classes:
	make_dir('{}/train/{}'.format(root_path, c))
	make_dir('{}/val/{}'.format(root_path, c))

# Copy train images
with open(img_list_path, 'r') as file:
	images_txt = file.readlines()
with open(split_path, 'r') as file:
	split_txt  = file.readlines()
for line in images_txt:
	indx = int(line.split('\n')[0].split(' ')[0])
	img_file = images_txt[indx - 1].split('\n')[0].split(' ')[1]
	if split_txt[indx - 1].split('\n')[0].split(' ')[1] == '1':
		print('{}. Copying {} to train.'.format(indx, img_file))
		shutil.copyfile(
			os.path.join(imgs_path, img_file),
			os.path.join(os.path.join(root_path, 'train'), img_file)
			)
	else:
		print('Copying {} to val.'.format(img_file))
		shutil.copyfile(
			os.path.join(imgs_path, img_file),
			os.path.join(os.path.join(root_path, 'val'), img_file)
			)

