import os
from PIL import Image, ImageOps

src_dir = "/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition"
tar_dir = "/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_augmented"


# Set the desired size for the resized images
target_size = (224, 224)

# Iterate through all files in the source directory
for dir in os.listdir(src_dir):
    if not os.path.exists(os.path.join(tar_dir, dir)):
        os.makedirs(os.path.join(tar_dir, dir))
    for trigger_type in os.listdir(os.path.join(src_dir, dir)):
        if not os.path.exists(os.path.join(tar_dir, dir, trigger_type)):
            os.makedirs(os.path.join(tar_dir, dir, trigger_type))
        for image in os.listdir(os.path.join(src_dir, dir, trigger_type)):
            image_path = os.path.join(src_dir, dir, trigger_type, image)
            target_path = os.path.join(tar_dir, dir, trigger_type, image)
            
            # Open the image using PIL
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            
            # Resize the image
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(target_path)
            
            print(f"Resized and saved: {target_path}")

print("Resizing and moving images completed.")