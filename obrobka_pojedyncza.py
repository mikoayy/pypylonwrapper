from pypylon_wraper import load_single_img
import matplotlib.pyplot as plt

img = load_single_img("/home/mikoay/Documents/niezdatne/h3.2_180_1/wiertlo_2025_06_18-12_35_50_004.bmp")

configs={
    "blur_ksize": (5,5),
    "blur_sigma": 0.0,
    "strenght": 1.5,
    "treshold": 35,
    "closing_kernel": (17,17),
    "max_workers": 4 
}

img_p = img.copy().pipeline(**configs)

plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
plt.title("orginał")
plt.imshow(img[0],cmap="gray")
plt.axis(False)

plt.subplot(1,2,2)
plt.title("przerobione")
plt.imshow(img_p[0],cmap="gray")
plt.axis(False)
plt.show()
