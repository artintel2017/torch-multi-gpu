# data_transfroms.py
# created: CS
# 各种数据变换
# 针对单个样本

'''
ElasticTransform: 
        弹性变换，根据扭曲场的平滑度与强度逐一地移动局部像素点实现模糊效果。
依赖：   albumentations

参数：
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
输入：
        numpy数组
返回：
        字典，形式为 {'image': array(...)}
示例：
        >>> import albumentations as A
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = Image.open('img.jpg')
        >>> np_img = np.asarray(img)
        >>> t = A.ElasticTransform()
        >>> img = Image.fromarray(t(image = np_img)['image'])
'''

'''
HueSaturationValue: 
        HSV对比度变换，通过向HSV空间中的每个像素添加或减少V值，修改色调和饱和度实现对比度转换。
依赖：   albumentations

参数：
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.
输入：
        numpy数组
返回：
        字典，形式为 {'image': array(...)}
示例：
        >>> import albumentations as A
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = Image.open('img.jpg')
        >>> np_img = np.asarray(img)
        >>> t = A.HueSaturationValue()
        >>> img = Image.fromarray(t(image = np_img)['image'])
'''

'''
IAASuperpixels: 
        超像素法，在最大分辨率处生成图像的若干个超像素，并将其调整到原始大小，再将原始图像中所有超像素区域按一定比例替换为超像素，其他区域不改变。
依赖：   albumentations
注意：   该方法可能速度较慢。
参数：
        p_replace (float): defines the probability of any superpixel area being replaced by the superpixel, i.e. by
            the average pixel color within its area. Default: 0.1.
        n_segments (int): target number of superpixels to generate. Default: 100.
        p (float): probability of applying the transform. Default: 0.5.
输入：
        numpy数组
返回：
        字典，形式为 {'image': array(...)}
示例：
        >>> import albumentations as A
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = Image.open('img.jpg')
        >>> np_img = np.asarray(img)
        >>> t = A.IAASuperpixels()
        >>> img = Image.fromarray(t(image = np_img)['image'])
'''

'''
IAAPerspective: 
        随机四点透视变换
依赖：   albumentations
参数：
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.
输入：
        numpy数组
返回：
        字典，形式为 {'image': array(...)}
示例：
        >>> import albumentations as A
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = Image.open('img.jpg')
        >>> np_img = np.asarray(img)
        >>> t = A.IAAPerspective()
        >>> img = Image.fromarray(t(image = np_img)['image'])
'''

'''
CoarseDropout: 
        在面积大小可选定、位置随机的矩形区域上丢失信息实现转换，产生黑色矩形块。
依赖：   albumentations
参数：
        max_holes (int): Maximum number of regions to zero out.
        max_height (int): Maximum height of the hole.
        max_width (int): Maximum width of the hole.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
        min_width (int): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
输入：
        numpy数组
返回：
        字典，形式为 {'image': array(...)}
示例：
        >>> import albumentations as A
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = Image.open('img.jpg')
        >>> np_img = np.asarray(img)
        >>> t = A.CoarseDropout()
        >>> img = Image.fromarray(t(image = np_img)['image'])
'''

'''
EdgeDetect: 
        边界检测，检测图像中的所有边缘，将它们标记为黑白图像，再将结果与原始图像叠加。
依赖：   imgaug
参数：
        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Blending factor of the edge image. At ``0.0``, only the original
        image is visible, at ``1.0`` only the edge image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
输入：
        numpy数组
返回：
        numpy数组
示例：
        >>> import imgaug.augmenters as iaa
        >>> import numpy as np
        >>> from PIL import Image 
        >>> img = Image.open('img.jpg')
        >>> np_img = np.asarray(img)
        >>> t = iaa.EdgeDetect(alpha=(0.0, 1.0))
        >>> img = Image.fromarray(t(image = np_img))
'''

'''
RandomAffine: 
        仿射变换，对图像进行旋转、水平偏移、裁剪、放缩等操作，保持中心不变。
依赖：   torchvision
参数：
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
输入：
        PIL图片
返回：
        PIL图片
示例：
        >>> from torchvision import transforms as T
        >>> from PIL import Image 
        >>> img = Image.open('img.jpg')
        >>> t = T.RandomAffine(25, translate=(0.2,0.2), scale=(0.8,1.2), shear=8, resample=Image.BILINEAR) 
        >>> img = t(img)
'''

'''
randomGausNoise:
        高斯模糊。
实现：
        def randomGausNoise(image):
            dice = random() 
            if dice<0.5:
                return image.filter(ImageFilter.GaussianBlur(radius=random()*1.7+0.5) )
            else: 
                return image
输入：
        PIL图片
返回：
        PIL图片
示例：
        >>> from torchvision import transforms as T
        >>> from random import random
        >>> from PIL import Image, ImageFilter
        >>> img = Image.open('img.jpg')
        >>> t = T.Lambda(randomGausNoise)
        >>> img = t(img)
'''

'''
cropAndPadImage:
        裁切图像并用黑色像素补齐图像至目标大小。
实现：
        target_size = (224, 224)
        background_color = (0,0,0)
        def cropAndPadImage(img):
            w, h = img.size
            if w==h:
                if w>target_size[0]:
                    return img.resize(target_size)
                else:
                    return img
            if w>h:
                x0 = int( (w-h)/4 )
                x1 = w - x0
                y0 = 0
                y1 = h
                padding_length = x1-x0
                padding_size = (padding_length, padding_length)
                pad_x0 = 0
                pad_x1 = padding_length
                pad_y0 = int( (w-h)/4 )
                pad_y1 = pad_y0 + h
            else :
                x0 = 0
                x1 = w
                y0 = int( (h-w)/4 )
                y1 = h - y0
                padding_length = y1-y0
                padding_size = (padding_length, padding_length)
                pad_x0 = int( (h-w)/4 )
                pad_x1 = pad_x0 + w
                pad_y0 = 0
                pad_y1 = padding_length
            cropped_img = img.crop( (x0,y0, x1,y1) )
            padded_img = Image.new('RGB', padding_size, background_color)
            #print(img.size, padding_size, cropped_img.size, (pad_x0, pad_y0, pad_x1, pad_y1) )
            padded_img.paste(cropped_img, (pad_x0, pad_y0, pad_x1, pad_y1) )
            resized_img = padded_img.resize(target_size)
            return resized_img
输入：
        PIL图片
返回：
        PIL图片
示例：
        >>> from torchvision import transforms as T
        >>> from random import random
        >>> from PIL import Image
        >>> img = Image.open('img.jpg')
        >>> t = T.Lambda(cropAndPadImage)
        >>> img = t(img)
'''