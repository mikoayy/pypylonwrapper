from pypylon_wraper import load_images, merge

configs={
    "blur_ksize": (5,5),
    "strenght": 1.5,
    "treshold": 35,
    "closing_kernel": (17,17),
    "max_workers": 16 
}
img_zdatne = load_images("/home/mikoay/Documents/zdatne",max_workers=16)
img_niezdatne = load_images("/home/mikoay/Documents/niezdatne",max_workers=16)
all_imgs = merge(img_zdatne,img_niezdatne)

print(img_zdatne.shape)
print(img_niezdatne.shape)
print(all_imgs.shape)

all_imgs.pipeline(**configs)
print(all_imgs.shape)

all_imgs.save(folder_path="/home/mikoay/Documents/przerobione")
