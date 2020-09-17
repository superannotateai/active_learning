import h5py
import numpy as np
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler

logger = getLogger("MainLogger")

def matrix_to_hdf5(f, matrix, hdf5_identifier_name):
    f.create_dataset(hdf5_identifier_name, data=matrix, compression="gzip")


def load_images_from_hdf5_file(file_path, image_or_slice_ids):
    """ Loads images from hdf5 file in shape (w, h, ch).
    Args:
        file_path (string) - Path to the hdf5 file.
        image_or_slice_ids (string) - if slices are not used, image_id, otherwise slice_id.
    """
    # logger.info("Loading images from hdf5 file {}.".format(file_path))
    X = []
    # Load valtrain
    with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
        for image_id in image_or_slice_ids:
            # logger.info("Loading image {}".format(image_id))
            im = np.array(f.get(image_id)) # (ch, w, h)
            im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
            im = np.swapaxes(im, 1, 2)  # -> (w, h, ch)
            X.append(im)
    return np.array(X)
