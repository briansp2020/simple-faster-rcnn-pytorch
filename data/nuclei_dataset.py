import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import cv2

from .util import read_image

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax

class NucleiDataset:
    """Bounding box dataset for 2018 Data Science Bowl.

    .. _`Nuclei`: https://www.kaggle.com/c/data-science-bowl-2018

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`NUCLEI_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/root/input/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, data_csv,
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        data_file_path = os.path.join(data_dir, data_csv)
        data_df = pd.read_csv(data_file_path)
        data_ids = []
        for i, row in data_df.iterrows():
            if row['ImageId'] != '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80': # bad labeling
                data_ids.extend([row['ImageId']] * int(row['mult_factor']))

        self.ids = data_ids
        self.data_dir = data_dir
        self.data_csv = data_csv
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = NUCLEI_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        #anno = ET.parse(
        #    os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        mask_path = os.path.join(self.data_dir, 'stage1_train_512', id_, 'masks')
        if os.path.exists(mask_path): # test data does not have masks
            mask_files = next(os.walk(mask_path))[2]
            for mask in mask_files:
                mask_filepath = os.path.join(mask_path, mask)
                mask_img = cv2.imread(mask_filepath, 0)
                bbox.append(bbox2(mask_img))
                label.append(NUCLEI_BBOX_LABEL_NAMES.index('nuclei'))
                difficult.append(0)
            # When `use_difficult==False`, all elements in `difficult` are False.
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'stage1_train_512', '%s/images/%s.png' %(id_, id_))
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example

NUCLEI_BBOX_LABEL_NAMES = (
    'nuclei',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label',
    'no-label')
