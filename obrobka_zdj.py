from pypylon_wraper import load_images, merge, get_photos_paths

configs={
    "blur_ksize": (5,5),
    "blur_sigma": 0.0,
    "strenght": 1.5,
    "treshold": 35,
    "closing_kernel": (17,17),
    "max_workers": 16
}

img_paths_new = get_photos_paths("/home/mikoay/Documents/zdatne")
img_paths_used = get_photos_paths("/home/mikoay/Documents/niezdatne")

img_new = load_images(img_paths_new,max_workers=16)
img_used = load_images(img_paths_used,max_workers=16)
all_imgs = merge(img_new,img_used)

print(img_new.shape)
print(img_used.shape)
print(all_imgs.shape)

all_imgs.pipeline(**configs)
print(all_imgs.shape)

all_imgs.save(folder_path="/home/mikoay/Documents/przerobione")
