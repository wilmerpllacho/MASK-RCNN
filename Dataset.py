import numpy as np
import albumentations as A
import cv2 as cv2
from skimage import exposure
import matplotlib.pyplot as plt
class Dataset(object):
    #se usa
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def clahe(self,image):
        img_clahe = exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)
        return img_clahe
    
    def load_lug(self, dataset):
        self.add_class("Lug", 1, "Izquierdo")
        self.add_class("Lug", 2, "Derecho")
        for info in dataset:
          height  = 512
          width   = 512
          self.add_image (
                source = "Lug",
                image_id=info['id'],
                path="",
                width=width,
                height=height,
                image=info['image'],
                mask=info['mask'])

    #se usa
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })
    def load_items(self,image_id):
      info = self.image_info[image_id]
      image = info['image']
      
      if image.ndim != 3:
          image_array = np.zeros((512, 512,3), dtype=np.float32)
          image_array[:,:,0] = image
          image_array[:,:,1] = image
          image_array[:,:,2] = image
          image = image_array
      #image = self.clahe(image)
      image = cv2.normalize(image, None , 0, 255, norm_type=cv2.NORM_MINMAX)

      label = self.image_info[image_id]['mask']
      hist,bins = np.histogram(image.flatten())
      cantidad_mask = 0 # Cantidad de pulmones
      if hist[5] > 0: # Existe pixeles pulmon izquierdo
          cantidad_mask = cantidad_mask + 1
      if hist[9] > 0: # Existe pixeles pulmon derecho
          cantidad_mask = cantidad_mask + 1
      mask = np.zeros([info['height'], info['width'], cantidad_mask],dtype=np.uint8)
      id_mask = np.ones([cantidad_mask], dtype=np.int32)
      posicion  = 0
      if hist[5] > 0: # Existe pixeles pulmon izquierdo
        Lug_Izquierdo = np.zeros((512, 512), dtype=np.float32)
        Lug_Izquierdo[:,:]  = np.where(label==0.5,255,Lug_Izquierdo[:,:]);
        mask[:,:,posicion] = Lug_Izquierdo
        id_mask[posicion]   = 1
        posicion = posicion + 1
      if hist[9] > 0: # Existe pixeles pulmon derecho   
        Lug_Derecho   = np.zeros((512, 512), dtype=np.float32)
        Lug_Derecho[:,:]  = np.where(label==1,255,Lug_Derecho[:, :]);
        mask[:,:,posicion] = Lug_Derecho
        id_mask[posicion]   = 2
      return image, mask.astype(np.bool), id_mask
    #se usa
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    #se usa
    def prepare(self, class_map=None):
        self.num_classes = len(self.class_info) # 2
        self.class_ids = np.arange(self.num_classes) # [0 1]
        #self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.class_names = [c["name"] for c in self.class_info] # [BG,LUG]
        self.num_images = len(self.image_info) # 2MIL
        self._image_ids = np.arange(self.num_images) # 2 MIL

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    @property
    def image_ids(self):
        return self._image_ids
