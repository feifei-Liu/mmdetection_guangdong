import mmcv

from ..registry import PIPELINES
from .compose import Compose


@PIPELINES.register_module
class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale, flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple)
        self.flip = flip


    def crop(self, col, row, img, img_shape, patch_size):
        w_temp = img_shape[1]//patch_size[0]
        h_temp = img_shape[0]//patch_size[1]
        x1 = row*w_temp
        y1 = col*h_temp
        x2 = row*w_temp + w_temp
        y2 = col*h_temp + h_temp
        #print(x1,y1,x2,y2)
        img_patch = img[y1:y2, x1:x2,:]
        img_shape = (h_temp, w_temp,3)
        return img_patch,img_shape

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                if 'col' in results.keys():
                    ## crop images
                    _results = results.copy()
                    img, img_shape = self.crop(_results['col'], _results['row'], _results['img'], _results['img_shape'],_results['patch_size'])
                    _results['img'] = img
                    _results['img_shape'] = img_shape
                    _results['ori_shape'] = img_shape
                    _results['scale'] = scale
                    _results['flip'] = flip
                    data = self.transforms(_results)
                    aug_data.append(data)
                else:
                    ## raw image
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    data = self.transforms(_results)
                    aug_data.append(data)

        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={}, img_scale={}, flip={})'.format(
            self.transforms, self.img_scale, self.flip)
        return repr_str
