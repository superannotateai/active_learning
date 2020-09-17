import numpy as np
import torch
from torch.utils import data
import h5py
import warnings
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler

# Logger
logger = getLogger("MainLogger")


if __name__ == '__main__':
    logger.addHandler(handler) 


class HDF5Dataset(data.Dataset):
    """Represents a HDF5 dataset. Loads images from compressed HDF5 file.
    
    Input params:
        image_file_path: Path to a HDF5 file containing all the image slices.
        mask_file_path: Path to a HDF5 file containing all the image masks.
        image_ids: List of strings with image or slice ids.
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, image_file_path, mask_file_path=None, image_ids=None, 
                 transform=None):
        super().__init__()
        self.image_file = h5py.File(image_file_path, 'r', libver='latest', swmr=True)
        # Sometimes we don't need to load the ground truth masks.
        if mask_file_path is None:
            self.mask_file = None
        else:
            self.mask_file = h5py.File(mask_file_path, 'r', libver='latest', swmr=True)

        self.image_ids = image_ids 
        self.transform = transform
            
    def __getitem__(self, index):
        # get data
        x = self.get_image(index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        if self.mask_file is None:
            return x, 0 # {'image': x, 'mask': 0/None}

        # get label
        y = self.get_mask(index)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.image_ids) 

    def get_mask(self, index):
        if self.mask_file==None:
            return None
        slice_id = self.image_ids[index]
        return self.load_ground_truth_mask(slice_id)

    def get_image(self, index):
        slice_id = self.image_ids[index]
        # print("Trying to load {}".format(slice_id))
        return self.load_image(slice_id)

    def load_image(self, slice_id):
        """ Loads image slice from hdf5 file in shape (w, h, ch).
        Args:
            slice_id (string) - if slices are not used, image_id, otherwise slice_id.
        """
        im = np.array(self.image_file.get(slice_id), dtype=np.float32) # (ch, w, h)
        # print("**************** Loaded image {} of shape {}".format(slice_id, im.shape))
        # NOTE(martun):  Ignore mean image for this time.
        #if self.mean_image is not None:
        #    im -= self.mean_image
        return im

    def load_ground_truth_mask(self, slice_id):
        mask = np.array(self.mask_file.get(slice_id))
        mask = (mask > 0.5).astype(np.uint8)
        #logger.info("Loaded a mask for image {} with {} filled pixels".format(
        #    image_id, str(np.sum(mask))))
        input_size = mask.shape[-1]
        mask = mask.reshape((1, input_size, input_size))
        return mask

