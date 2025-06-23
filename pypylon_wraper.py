import numpy as np
import cv2
import concurrent.futures
from pypylon import pylon
from functools import partial
from pathlib import Path
from typing import Union
from time import sleep
import time
from PIL import Image
from typing import Optional

class Images():
    """
    Klasa do zarządzania kolekcją obrazów z podstawowymi operacjami przetwarzania.
    
    Obsługuje obrazy w formatach BGR i Grayscale, umożliwia zapisywanie, 
    ładowanie oraz podstawowe transformacje obrazów.
    
    Attributes:
        images (np.ndarray): Tablica numpy zawierająca obrazy
        format (str): Format obrazów ("BGR" lub "Gray")
        shape (tuple): Kształt tablicy obrazów
    """
    
    def __init__(self, images: np.ndarray, format: str):
        """
        Inicjalizuje obiekt Images.
        
        Args:
            images (np.ndarray): Tablica numpy z obrazami
            format (str): Format obrazów ("BGR" lub "Gray")
        """
        self.images = images
        self.format = format
        self.shape = self.images.shape
        
    def __getitem__(self, x):
        return self.images[x]
    
    def __iter__(self):
        return iter(self.images)
    
    def copy(self):
        return Images(self.images.copy(), self.format)
    
        
    def save(self, folder_path: Optional[str] = None, filename: str = "saved_image", format: str = "bmp"): 
        """
        Zapisuje obrazy do plików w określonym formacie.
        
        Args:
            folder_path (Optional[str]): Ścieżka do folderu docelowego. 
                                       Jeśli None, używa "dummy_folder"
            filename (str): Nazwa bazowa pliku (bez rozszerzenia)
            format (str): Format zapisu ("png", "bmp" lub "npy")
            
        Returns:
            Path: Ścieżka do utworzonego folderu
            
        Raises:
            TypeError: Gdy format nie jest "png", "bmp" ani "npy"
        """
        if folder_path is None: folder_path = "dummy_folder"
        if format not in("png","npy", "bmp"): raise TypeError("invalid format use 'png', 'npy' or bmp")
        path = self._create_folder(folder_path)
        if format in ("bmp", "png"):
            
            if len(self.images) > 1:
                for i, img in enumerate(self.images):
                    img = Image.fromarray(img)
                    time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
                    img_filename = path / f"{filename}_{time_stamp}_{i:03d}.{format}"
                    img.save(str(img_filename))
                    
            else :
                imgs = np.squeeze(self.images,0)
                imgs = Image.fromarray(imgs)
                time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
                img_filename:Path = path / f"{filename}_{time_stamp}.{format}"
                imgs.save(str(img_filename))
                
        if format == "npy":
            time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
            img_filename = path / f"{filename}_{time_stamp}"
            np.save(str(img_filename),self.images)
        return path
    
    def pipeline(self,blur_ksize:tuple=(5,5),blur_sigma:float=0.0,strenght:float=1.5,
                 treshold:int=35,closing_kernel: tuple=(17,17),max_workers=4):
        """
        Przetwarza obrazy przez pipeline obróbki: blur → sharpening → grayscale → threshold → morfologia.
        
        Wykorzystuje wielowątkowość do równoległego przetwarzania obrazów.
        
        Args:
            blur_ksize (tuple): Rozmiar kernela dla Gaussian Blur (domyślnie (5,5))
            blur_sigma (float): Odchylenie standardowe dla Gaussian Blur (domyślnie 0.0)
            strenght (float): Siła wyostrzania (domyślnie 1.5)
            treshold (int): Próg binaryzacji (domyślnie 35)
            closing_kernel (tuple): Rozmiar kernela dla operacji morfologicznej (domyślnie (17,17))
            max_workers (int): Maksymalna liczba wątków (domyślnie 4)
            
        Returns:
            self: Zwraca obiekt Images z przetworzonymi obrazami (modyfikuje siebie)
            
        Note:
            Pipeline kolejno wykonuje: rozmycie → wyostrzanie → konwersja na grayscale →
            progowanie binarne → zamknięcie morfologiczne
        """
        
        def photo_processor(img,ksize,sigma,strenght,treshold,closing_kernel):
            blur = cv2.GaussianBlur(img,ksize,sigma)
            sharp = cv2.addWeighted(blur,1.0+strenght,blur,-strenght,0)
            gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            _, tresh = cv2.threshold(gray,treshold,255,cv2.THRESH_BINARY)
            kernel = np.ones(closing_kernel,np.uint8)
            closing = cv2.morphologyEx(tresh,cv2.MORPH_CLOSE,kernel)
            return closing
        # def photo_processor(img,ksize,strenght,treshold,closing_kernel):
        #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #     _, tresh = cv2.threshold(gray,treshold,255,cv2.THRESH_BINARY)
        #     kernel = np.ones(closing_kernel,np.uint8)
        #     closing = cv2.morphologyEx(tresh,cv2.MORPH_CLOSE,kernel)
        #     blur = cv2.GaussianBlur(closing,ksize,0)
        #     sharp = cv2.addWeighted(blur,1.0+strenght,blur,-strenght,0)
        #     return sharp
        
        imgs = self.images
        processor = partial(photo_processor,ksize=blur_ksize,sigma=blur_sigma,strenght=strenght,
                            treshold=treshold,closing_kernel=closing_kernel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as excecutor:
            procesed_imgs = list(excecutor.map(processor,imgs))
            procesed_imgs = np.array(procesed_imgs)
            self.images = procesed_imgs
            self.shape = self.images.shape
            self.format = "GRAY"
        return self
            
        
    
    def change_format(self, format: str):
        """
        Zmienia format obrazów między BGR a Grayscale.
        
        Args:
            format (str): Docelowy format ("Gray" lub "BGR")
            
        Returns:
            Images: Obiekt Images z obrazami w nowym formacie (modyfikuje siebie)
            
        Raises:
            TypeError: Gdy próbuje się zmienić na ten sam format
        """
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
            
    def _create_folder(self, folder_path: str):
        """
        Tworzy folder jeśli nie istnieje.
        
        Args:
            folder_path (str): Ścieżka do folderu
            
        Returns:
            Path: Obiekt Path do utworzonego folderu
            
        Raises:
            TypeError: Gdy folder_path nie jest stringiem
        """
        if isinstance(folder_path,str):
            fpath = Path(folder_path)
            fpath.mkdir(parents=True,exist_ok=True)
            return fpath
        else:
            raise TypeError("folder path must be a string")      
        
class PypylonWrapper:
    """
    Wrapper dla biblioteki pypylon umożliwiający łatwą obsługę kamer Basler.
    
    Klasa zapewnia uproszczony interfejs do grabowania obrazów, 
    konfiguracji kamery oraz obsługi live video.
    
    Attributes:
        cam: Obiekt kamery pypylon.InstantCamera
        format: Format pikseli kamery (jeśli skonfigurowany)
    """

    def __init__(self, configs: Optional[dict] = None, pfs_file_path: Optional[str] = None):
        """
        Inicjalizuje wrapper kamery Basler.
        
        Args:
            configs (Optional[dict]): Słownik z konfiguracją kamery.
                                    Możliwe klucze: 'img_size', 'gain', 
                                    'pixel_format', 'frame_rate'
            pfs_file_path (Optional[str]): Ścieżka do pliku konfiguracyjnego .pfs 
                                         z ustawieniami kamery
        """
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        if configs:
            self.cam.Open()
            self._cam_configs(**configs)
            self.cam.Close()
            
        if pfs_file_path:
            pfs_path = Path(pfs_file_path)
            if pfs_path.suffix.lower() != ".pfs": raise TypeError("file must be .pfs")
            self.cam.Open()
            pylon.FeaturePersistence.Load(str(pfs_path),self.cam.GetNodeMap())
            self.cam.Close()
        
        
    def _cam_configs(self, img_size: Optional[tuple[int,int]] = None,
                     gain: Optional[float] = None,
                     pixel_format: Optional[str] = None,
                     frame_rate: Optional[int] = None):
        """
        Konfiguruje parametry kamery.
        
        Args:
            img_size (Optional[tuple[int,int]]): Rozmiar obrazu (szerokość, wysokość)
            gain (Optional[float]): Wzmocnienie kamery
            pixel_format (Optional[str]): Format pikseli
            frame_rate (Optional[int]): Liczba klatek na sekundę
        """
        if img_size:
            width, height = img_size
            self.cam.Width.Value = width
            self.cam.Height.Value = height
        if gain:
            self.cam.Gain.Value = gain
        if pixel_format:
            self.cam.PixelFormat.Value = pixel_format
            self.format = pixel_format
        if frame_rate:
            self.cam.AcquisitionFrameRate.Value = frame_rate
        
            
    def grab_images(self, num_of_images: int = 5, format: str = "Gray", wait: Optional[int|float] = None):
        """
        Grabuje określoną liczbę obrazów z kamery.
        
        Args:
            num_of_images (int): Liczba obrazów do pobrania
            format (str): Format obrazów ("Gray" lub "BGR")
            wait (Optional[int|float]): Czas oczekiwania między obrazami w sekundach
            
        Returns:
            Images: Obiekt Images z pobranymi obrazami
            
        Raises:
            TypeError: Gdy format nie jest "Gray" ani "BGR"
        """
        
        if format not in ("Gray","BGR"): raise TypeError("invalid format use Gray or BGR")
        converter = None 
        if format == "BGR": 
            converter = self._BGR_converter()
        
        imgs=[] 
        timeout = 2000
        if wait and timeout/1000 <= wait: timeout = 2000 + wait*1000
        self.cam.Open()
        self.cam.StartGrabbing()
        
        for i in range(num_of_images):
            with self.cam.RetrieveResult(timeout) as result:
                result = converter.Convert(result) if converter else result
                imgs.append(result.Array)
            if wait: sleep(wait)
       
        imgs = np.array(imgs)
        return Images(imgs,format)
    
    def auto_grabing(self, imgs_to_grab: Optional[int] = None, wait: Optional[int|float] = None):
        """
        Wyświetla live video z kamery z możliwością grabowania obrazów.
        
        Sterowanie:
        - 'q': Zakończenie
        - Spacja: Rozpoczęcie grabowania (jeśli imgs_to_grab jest ustawione)
        
        Args:
            imgs_to_grab (Optional[int]): Liczba obrazów do pobrania po naciśnięciu spacji
            wait (Optional[int|float]): Czas oczekiwania między grabowaniem obrazów
            
        Returns:
            Optional[Images]: Obiekt Images z pobranymi obrazami lub None
        """
        grabbing_flag = False
        i = imgs_to_grab
        imgs_grabed = 0
        end_time = 0 
        imgs = []
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = self._BGR_converter()
        
        while self.cam.IsGrabbing():
            timeN = time.time()
            grabresult = self.cam.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)
            
            if grabresult.GrabSucceeded():
                image = converter.Convert(grabresult)
                image = image.Array
            
                if grabbing_flag and i is not None and imgs_grabed < i:
                        if wait and timeN - end_time > wait:
                            imgs.append(image)
                            imgs_grabed += 1
                            end_time = time.time()
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
                
            else:
                print("grabbing failed")
                break
                
    def button_grabbing(self,button: str = "h"):
        """
        Wyświetla live video z kamery z możliwością grabowania obrazów poprzez naciśnięcie klawisza.
        
        Sterowanie:
        - 'q': Zakończenie i zwrócenie pobranych obrazów
        - button (domyślnie 'h'): Pobranie pojedynczego obrazu
        
        Args:
            button (str): Klawisz do grabowania obrazów (domyślnie "h")
            
        Returns:
            Optional[Images]: Obiekt Images z pobranymi obrazami lub None jeśli nie pobrano żadnych
        """
        num_gr_photos = 0
        grabed_photos = []
        
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = self._BGR_converter()
        while self.cam.IsGrabbing():
            grabresult = self.cam.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)
            
            if grabresult.GrabSucceeded():
                image = converter.Convert(grabresult)
                image = image.Array
                cv_img = np.array(image).copy()
            else:
                print("grabbing failed")
                break
            
            cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
            cv_img = cv2.resize(cv_img,(1280,720))
            cv_img = cv2.putText(cv_img,f"photos: {num_gr_photos}", (40,50),cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=1,color=(0,255,0),thickness=2)
            
            cv2.imshow("BaslerCam",cv_img)
            input = cv2.waitKey(1)
            if input == ord("q"):
                    cv2.destroyAllWindows()
                    if num_gr_photos > 0:
                        grabed_photos = np.array(grabed_photos)
                        return Images(grabed_photos,"BGR")
                    else: break
                
            if input == ord(button):
                grabed_photos.append(image)
                num_gr_photos += 1
            
                
    
    def _BGR_converter(self):
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return converter
    
    def close_cam(self):
        """
        Zamyka kamerę i kończy grabowanie.
        
        Należy wywołać tę metodę po zakończeniu pracy z kamerą.
        """
        self.cam.StopGrabbing()
        self.cam.Close()


def load_single_img(path: str):
    """
    Ładuje pojedynczy obraz z podanej ścieżki.
    
    Args:
        path (str): Ścieżka do pliku obrazu
        
    Returns:
        Images: Obiekt Images z pojedynczym obrazem w formacie BGR
        
    Note:
        Obraz jest automatycznie rozszerzany o wymiar batch (axis=0)
    """
    img = cv2.imread(path)
    img = np.array(img)
    img = np.expand_dims(img,axis=0)    
    return Images(img,"BGR")

def merge(a1, a2):
    """
    Łączy dwa obiekty Images w jeden.
    
    Args:
        a1 (Images): Pierwszy obiekt Images
        a2 (Images): Drugi obiekt Images
        
    Returns:
        Images: Połączony obiekt Images
        
    Raises:
        TypeError: Gdy formaty obrazów się różnią
    """
    if a1.format != a2.format: raise TypeError("invalid format")
    return Images(np.concatenate((a1.images,a2.images)),a1.format)

def get_photos_paths(folder_path: str):
    """
    Pobiera ścieżki do wszystkich obrazów z podfoldedów.
    
    Przeszukuje wszystkie podfoldery w podanej ścieżce i zbiera 
    ścieżki do plików .png i .bmp.
    
    Args:
        folder_path (str): Ścieżka do folderu głównego
        
    Returns:
        np.ndarray: Tablica ze ścieżkami do obrazów
    """
    img_paths = []
    folder__path=Path(folder_path)
    sub_f_paths = [subFpath for subFpath in folder__path.iterdir()]
    for subFpath in sub_f_paths:
        for imgPath in subFpath.iterdir():
            if imgPath.suffix.lower() in (".png",".bmp"):
                img_paths.append(str(imgPath))
    return np.array(img_paths)

def load_images_npy(path: str):
    """
    Ładuje obrazy z pliku .npy.
    
    Args:
        path (str): Ścieżka do pliku .npy
        
    Returns:
        Images: Obiekt Images z załadowanymi obrazami
        
    Note:
        Format jest wykrywany automatycznie na podstawie liczby kanałów obrazu
    """
    fpath = Path(path)
    if fpath.suffix.lower() == ".npy":
        imgs = np.load(fpath,allow_pickle=True)
        if len(imgs) > 0 and all(len(img.shape) > 0 and img.shape[-1] == 3 for img in imgs):
            format = "BGR"
        else: format = "Gray"
        return Images(imgs,format) 

def load_images(paths_list: Union[list[str],np.ndarray], max_workers: int = 4):
    """
    Ładuje obrazy równolegle z podanych ścieżek.
    
    Wykorzystuje wielowątkowość do szybkiego ładowania dużej liczby obrazów.
    
    Args:
        paths_list (Union[list[str], np.ndarray]): Lista lub tablica ścieżek do obrazów
        max_workers (int): Maksymalna liczba wątków (domyślnie 4)
        
    Returns:
        Images: Obiekt Images z załadowanymi obrazami w formacie BGR
        
    Note:
        Wszystkie obrazy są ładowane jako BGR niezależnie od oryginalnego formatu
    """
    
    def _load_single_img(imgPath: str):
       return cv2.imread(imgPath)
   
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        imgs = list(executor.map(_load_single_img,paths_list))
    imgs = np.array(imgs)
    return Images(imgs,format="BGR") 


    
