import os

# Prepare dataset
os.system('python ./prepare_dataset.py --images_dir "../data/Training and Validation/VDSR/Dataset" --output_dir "../data/Training and Validation/VDSR/Scale 2/train" --image_size 42 --step 42 --scale 2 --num_workers 10')

#os.system('python ./prepare_dataset.py --images_dir "../data/Training and Validation/VDSR/Dataset" --output_dir "../data/Training and Validation/VDSR/Scale 3/train" --image_size 42 --step 42 --scale 3 --num_workers 10')

os.system('python ./prepare_dataset.py --images_dir "../data/Training and Validation/VDSR/Dataset" --output_dir "../data/Training and Validation/VDSR/Scale 4/train" --image_size 44 --step 44 --scale 4 --num_workers 10')

# Split train and valid
os.system('python ./split_train_valid_dataset.py --train_images_dir "../data/Training and Validation/VDSR/Scale 2/train" --valid_images_dir "../data/Training and Validation/VDSR/Scale 2/valid" --valid_samples_ratio 0.1')

#os.system('python ./split_train_valid_dataset.py --train_images_dir "../data/Training and Validation/VDSR/Scale 3/train" --valid_images_dir "../data/Training and Validation/VDSR/Scale 3/valid" --valid_samples_ratio 0.1')

os.system('python ./split_train_valid_dataset.py --train_images_dir "../data/Training and Validation/VDSR/Scale 4/train" --valid_images_dir "../data/Training and Validation/VDSR/Scale 4/valid" --valid_samples_ratio 0.1')
