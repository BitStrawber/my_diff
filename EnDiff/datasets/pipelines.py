import cv2
import mmcv
import copy
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile, Resize, DefaultFormatBundle, to_tensor
from mmcv.parallel import DataContainer as DC

@PIPELINES.register_module()
class LoadLqHqImages(LoadImageFromFile):
    def __call__(self, results):
        results = super().__call__(results)
        hq_img_filename = results['hq_img_filename']
        hq_img_img_bytes = self.file_client.get(hq_img_filename)
        hq_img = mmcv.imfrombytes(hq_img_img_bytes, flag=self.color_type, channel_order=self.channel_order)
        
        results['hq_img'] = hq_img
        results['img_fields'] = ['hq_img', 'img']
        return results

@PIPELINES.register_module()
class ResizeLqHqImages(Resize):
    def __call__(self, results):
        fields = copy.deepcopy(results['img_fields'])
        resize_fields = []
        for f in results['img_fields']:
            if f != 'hq_img':
                resize_fields.append(f)
        results['img_fields'] = resize_fields

        results = super().__call__(results)
        hq_img, w_scale, h_scale = mmcv.imresize_like(
            results['hq_img'],
            results['img'],
            return_scale=True,
            interpolation=self.interpolation,
            backend=self.backend)
        results['hq_img'] = hq_img
        results['img_fields'] = fields
        return results

@PIPELINES.register_module()
class FormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        results = super().__call__(results)
        hq_img = results['hq_img']
        if self.img_to_float is True and hq_img.dtype == np.uint8:
            hq_img = hq_img.astype(np.float32)
        if len(hq_img.shape) < 3:
            hq_img = np.expand_dims(hq_img, -1)
        if not hq_img.flags.c_contiguous:
            hq_img = np.ascontiguousarray(hq_img.transpose(2, 0, 1))
            hq_img = to_tensor(hq_img)
        else:
            hq_img = to_tensor(hq_img).permute(2, 0, 1).contiguous()
            results['hq_img'] = DC(
                    hq_img, padding_value=self.pad_val['img'], stack=True)
        return results


@PIPELINES.register_module()
class CropPadding:
    """兼容MMDetection原版Pad的裁剪类"""

    def __init__(self, keys=('img',)):
        self.keys = keys

    def __call__(self, results):
        # 关键：通过pad_shape和img_shape计算填充量
        if 'pad_shape' not in results or 'img_shape' not in results:
            return results

        original_h, original_w = results['img_shape'][:2]
        padded_h, padded_w = results['pad_shape'][:2]

        # 计算右下角填充量（原版Pad的填充逻辑）
        pad_bottom = padded_h - original_h
        pad_right = padded_w - original_w

        # 执行裁剪（去除右下填充部分）
        for key in self.keys:
            if key in results:
                if isinstance(results[key], np.ndarray):
                    results[key] = results[key][:original_h, :original_w]
                elif hasattr(results[key], 'crop'):  # 处理mask
                    results[key] = results[key].crop(
                        np.array([0, 0, original_w, original_h]))

        # 更新元信息
        results['img_shape'] = (original_h, original_w)
        results['pad_shape'] = results['img_shape']  # 裁剪后两者一致
        return results