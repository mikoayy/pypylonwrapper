from pypylon_wraper import PypylonWrapper
basler = PypylonWrapper(pfs_file_path="/home/mikoay/Documents/pypylon/a2A1920-160ucPRO_40436060.pfs")

loop = True

while loop:
    folder_name = input("Enter folder name to save images: ")
    if folder_name == "exit":
        loop = False
        print("Exiting...")
        break
        
    a = basler.button_grabbing()
    basler.close_cam()
    if a is not None:
        a.save(folder_path=f"/home/mikoay/Documents/zdatne/{folder_name}", filename="wiertlo")
        print(a.images.shape)
    else: print("no images captured")


