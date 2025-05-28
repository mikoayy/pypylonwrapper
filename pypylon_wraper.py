import numpy as np
import uuid
import cv2
from pypylon import pylon
from pathlib import Path
from time import sleep
from time import time
from PIL import Image
from typing import Optional

class Images():
    def __init__(self,images: np.ndarray,format:str):
        self.images = images
        self.format = format
        self.shape = images.shape
        
    def __getitem__(self,x):
        return self.images[x]
    
    def __iter__(self):
        return iter(self.images)
        
    def save(self,folder_path: Optional[str] = None,filename: str ="saved_image",format: str="png"): 
        if folder_path is None: folder_path = "dummy_folder"
        if format not in("png","npy"): raise TypeError("invalid format use 'png' or 'npy'")
        path = self._create_folder(folder_path)
        if format == "png":
            
            if len(self.images) > 1:
                for img in self.images:
                    img = Image.fromarray(img)
                    id = uuid.uuid1()
                    img_filename = path / f"{filename}_{id}.png"
                    img.save(str(img_filename))
            else :
                imgs = np.squeeze(self.images,0)
                imgs = Image.fromarray(imgs)
                id = uuid.uuid1()
                img_filename:Path = path / f"{filename}_{id}.png"
                imgs.save(str(img_filename))
                
        if format == "npy":
            id = uuid.uuid1()
            img_filename = path / f"{filename}_{id}"
            np.save(str(img_filename),self.images)
        return path
    
    def pipline(self,treshold: int = 150):
        images = self.images
        format = self.format
        bimgs = []
        if format == "BGR":
            images = np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images])
            self.format = "Gray"
        
        for img in images:
            _, bimg =cv2.threshold(img,treshold,255,cv2.THRESH_BINARY)
            bimgs.append(bimg)
        bimgs = np.array(bimgs)
        self.images = bimgs
        return self
    
    def change_format(self,format:str):
        saved_format = self.format
        images = self.images
        if saved_format == format: raise TypeError("Can't change to the same format")
        elif format == "Gray":
            g_imgs = np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images])
            self.images = g_imgs
            self.shape = g_imgs.shape
            self.format = "Gray"
        elif format == "BGR":
            bgr_imgs = np.array([cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) for img in images])
            self.images = bgr_imgs
            self.shape = bgr_imgs.shape
            self.format = "BGR"
        return self
            
    def _create_folder(self, folder_path:str):
        if isinstance(folder_path,str):
            fpath = Path(folder_path)
            fpath.mkdir(parents=True,exist_ok=True)
            return fpath
        else:
            raise TypeError("folder path must be a string")   
        
def load_images(path:str):
    fpath = Path(path)
    if fpath.suffix.lower == ".npy":
        imgs = np.load(fpath,allow_pickle=True)
        if len(imgs) > 0 and all(len(img.shape) > 0 and img.shape[-1] == 3 for img in imgs):
            format = "BGR"
        else: format = "Gray"
        return Images(imgs,format)        
        
    imgs = []
    for img_path in fpath.iterdir():
        if img_path.suffix.lower() == ".png":
           img = cv2.imread(str(img_path))
           imgs.append(img)
           
    imgs = np.array(imgs)
    if len(imgs) > 0 and all(len(img.shape) > 0 and img.shape[-1] == 3 for img in imgs):
        format = "BGR"
    else: format = "Gray"
    return Images(imgs,format)        
        
class PypylonWrapper:
    def __init__(self,configs:Optional[dict] = None):
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        if configs:
            self.cam.Open()
            self._cam_configs(**configs)
            self.cam.Close()
        
        
    def _cam_configs(self,img_size: Optional[tuple[int,int]] = None,
                     gain: Optional[float] = None,
                     pixel_format: Optional[str] = None,
                     frame_rate: Optional[int] = None):
        if img_size:
            width, height = img_size
            self.cam.Width.Value = width
            self.cam.Height.Value = height
        if gain:
            self.cam.Gain.Value = gain
        if pixel_format:
            self.cam.PixelFormat.Value = pixel_format
        if frame_rate:
            self.cam.AcquisitionFrameRate.Value = frame_rate
        
            
    def grab_images(self,num_of_iamges: int = 5,format: str = "grayscale",wait: Optional[int|float] = None):
        
        if format not in ("grayscale","BGR"): raise TypeError("invalid format use grayscale or BGR")
        converter = None 
        if format == "BGR": 
            converter = self._BGR_conventer()
        
        imgs=[] 
        timeout = 2000
        if wait and timeout/1000 <= wait: timeout = 2000 + wait*1000
        self.cam.Open()
        self.cam.StartGrabbing()
        
        for i in range(num_of_iamges):
            with self.cam.RetrieveResult(timeout) as result:
                result = converter.Convert(result) if converter else result
                imgs.append(result.Array)
            if wait: sleep(wait)
       
        imgs = np.array(imgs)
        return Images(imgs,format)
    
    def live_video(self, imgs_to_grab: Optional[int] = None, wait: Optional[int|float] = None):
        grabbing_flag = False
        i = imgs_to_grab
        imgs_grabed = 0
        end_time = 0 
        imgs = []
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = self._BGR_conventer()
        
        while self.cam.IsGrabbing():
            timeN = time()
            grabresult = self.cam.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)
            
            if grabresult.GrabSucceeded():
                image = converter.Convert(grabresult)
                image = image.Array
                
                if grabbing_flag and i is not None and imgs_grabed < i:
                        if wait and timeN - end_time > wait:
                            imgs.append(image)
                            imgs_grabed += 1
                            end_time = time()
                        if not wait: 
                            imgs.append(image)
                            imgs_grabed += 1
                            
                    
                if i is not None and imgs_grabed >= i:
                    imgs = np.array(imgs)
                    return Images(imgs,"BGR")
                
                cv2.imshow("BaslerCam",image)
                
                input = cv2.waitKey(1)
                if input == ord("q"):
                    cv2.destroyAllWindows()
                    break
                if input == ord(" ") and imgs_to_grab is not None: grabbing_flag = True

    
    def _BGR_conventer(self):
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return converter
    
    def close_cam(self):
        self.cam.StopGrabbing()
        self.cam.Close()
        