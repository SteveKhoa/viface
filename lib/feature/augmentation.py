import albumentations
import cv2
import datetime
import os

directory_current = os.getenv("DIRECTORY_CURRENT")

def create_augmented_images(cv2_image_imread):
    face_image = cv2.cvtColor(cv2_image_imread, cv2.COLOR_BGR2RGB)

    # Define all the augmentations
    blur = albumentations.Blur(blur_limit=(7,13), p=1)
    random_brightness = albumentations.RandomBrightnessContrast(brightness_limit=0.6, p=1)
    random_contrast = albumentations.RandomBrightnessContrast(contrast_limit=0.9, p=1)
    horizontal_flip = albumentations.HorizontalFlip(p=1)
    affine = albumentations.Affine(scale=[0.5,2], translate_percent=[-0.05,0.05], rotate=[-45, 45], shear=[-15,-15], p=1)
    cutout = albumentations.Cutout(num_holes=50, max_h_size=20, max_w_size=20, fill_value=255, p=1.0)

    # Run augmentations
    face_image_blur = blur(image=face_image)
    face_image_random_brightness = random_brightness(image=face_image)
    face_image_random_contrast = random_contrast(image=face_image)
    face_image_horizontal_flip = horizontal_flip(image=face_image)
    face_image_affine = affine(image=face_image)
    face_image_cutout = cutout(image=face_image)
    face_image_cutout_2 = cutout(image=face_image)
    face_image_cutout_3 = cutout(image=face_image)

    face_image_blur = face_image_blur["image"]
    face_image_random_brightness = face_image_random_brightness["image"]
    face_image_random_contrast = face_image_random_contrast["image"]
    face_image_horizontal_flip = face_image_horizontal_flip["image"]
    face_image_affine = face_image_affine["image"]
    face_image_cutout = face_image_cutout["image"]
    face_image_cutout_2 = face_image_cutout_2["image"]
    face_image_cutout_3 = face_image_cutout_3["image"]

    # Pre-log transformation
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image_blur = cv2.cvtColor(face_image_blur, cv2.COLOR_BGR2RGB)
    face_image_random_brightness = cv2.cvtColor(face_image_random_brightness, cv2.COLOR_BGR2RGB)
    face_image_random_contrast = cv2.cvtColor(face_image_random_contrast, cv2.COLOR_BGR2RGB)
    face_image_horizontal_flip = cv2.cvtColor(face_image_horizontal_flip, cv2.COLOR_BGR2RGB)
    face_image_affine = cv2.cvtColor(face_image_affine, cv2.COLOR_BGR2RGB)
    face_image_cutout = cv2.cvtColor(face_image_cutout, cv2.COLOR_BGR2RGB)
    face_image_cutout_2 = cv2.cvtColor(face_image_cutout_2, cv2.COLOR_BGR2RGB)
    face_image_cutout_3 = cv2.cvtColor(face_image_cutout_3, cv2.COLOR_BGR2RGB)

    # Logs the image
    now = datetime.datetime.now()
    dir_target = f"{directory_current}/logs/images_augmented_{now}"
    os.mkdir(dir_target)

    cv2.imwrite(f"{dir_target}/face_image.jpg", face_image)
    cv2.imwrite(f"{dir_target}/face_image_blur.jpg", face_image_blur)
    cv2.imwrite(f"{dir_target}/face_image_random_brightness.jpg", face_image_random_brightness)
    cv2.imwrite(f"{dir_target}/face_image_random_contrast.jpg", face_image_random_contrast)
    cv2.imwrite(f"{dir_target}/face_image_horizontal_flip.jpg", face_image_horizontal_flip)
    cv2.imwrite(f"{dir_target}/face_image_affine.jpg", face_image_affine)
    cv2.imwrite(f"{dir_target}/face_image_cutout.jpg", face_image_cutout)
    cv2.imwrite(f"{dir_target}/face_image_cutout_2.jpg", face_image_cutout_2)
    cv2.imwrite(f"{dir_target}/face_image_cutout_3.jpg", face_image_cutout_3)

    return [
        face_image,
        face_image_blur,
        face_image_random_brightness,
        face_image_random_contrast,
        face_image_horizontal_flip,
        face_image_affine,
        face_image_cutout,
        face_image_cutout_2,
        face_image_cutout_3,
    ]