import cv2
import numpy as np
from skimage.io import imread,imsave
from skimage.transform import resize

def resize_to_width(img, new_width):
    height, width = img.shape[:2]
    new_height = int(new_width * 1.0 / width * height)
    newsize = (new_width, new_height)
    return cv2.resize(img,  newsize)


def resize_square_rgb(img, new_width, nchannels=3):
    if img.shape[0] == new_width and img.shape[1] == new_width:
        return img[:,:,0:nchannels]
    else:
        return resize(img[:,:,0:nchannels], (new_width, new_width, img.shape[2]), preserve_range=True)


def crop_to_square(img):
    if img.shape[0] == img.shape[1]:
        return img

    marg = abs(img.shape[0] - img.shape[1])
    lmarg = int(marg/2)
    rmarg = int(marg - lmarg)
    if img.shape[0] > img.shape[1]:
        if rmarg > 0:
            return img[lmarg:-rmarg, :, :]
        else:
            return img[lmarg:, :, :]
    else:
        if rmarg > 0:
            return img[:, lmarg:-rmarg, :]
        else:
            return img[:, lmarg:, :]


def read_resize_square(filename, width, dtype=np.float32, nchannels=3):
    img = imread(filename)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3 and nchannels == 3:
        img = img[:,:,0:3]
    elif img.shape[2] == 1:
        img = np.tile(img, [1,1,nchannels])
    img_sq = crop_to_square(img)
    if width > 0:
        img_resized = resize_square_rgb(img_sq, width, nchannels=nchannels)
    else:
        img_resized = img_sq

    if dtype == np.float32:
        return img_resized.astype(np.float32) / 255.0
    else:
        return img_resized.astype(dtype)


def concat_images(images, images_per_row=None):
    '''
    Concatenates images; must already be size-compatible.
    '''
    per_row = images_per_row
    if per_row is None:
        per_row = len(images)

    composite = None
    per_row = int(per_row)
    nrows = len(images) / per_row
    leftover = len(images) - nrows * per_row
    for i in range(nrows):
        tmp = np.concatenate(images[per_row * i : per_row * i + per_row], axis=1)
        if composite is None:
            composite = tmp
        else:
            composite = np.concatenate((composite, tmp), axis=0)
    if leftover > 0:
        mockimg = np.zeros(images[0].shape, dtype=np.uint8)
        mockimg.fill(255)
        tmp = np.concatenate(
            images[-leftover:] + [mockimg for x in range(per_row - leftover)],
            axis=1)
        if composite is None:
            composite = tmp
        else:
            composite = np.concatenate((composite, tmp), axis=0)
    return composite


def read_all_frames(video_file):
    vidcap = cv2.VideoCapture(video_file)
    frames = []
    success = True
    while success:
        success, frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


def sum_channels_absdiff(im0, im1):
    d = cv2.absdiff(im0, im1)
    r,g,b = cv2.split(d)
    dsum = cv2.add(r, cv2.add(g, b))
    return dsum


def img_to_true_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    # Note, in OpenCV: L = L * 255/100 ; a = a + 128 ; b = b + 128
    alab = np.zeros(img.shape, dtype=np.float)
    alab[:,:,0] = lab[:,:,0].astype(dtype=np.float) * 100.0 / 255;
    alab[:,:,1] = lab[:,:,1].astype(dtype=np.float) - 128;
    alab[:,:,2] = lab[:,:,2].astype(dtype=np.float) - 128;
    return alab


def create_checkered_image(img_width, patch_width, color1=[0.9, 0.9, 0.9, 1.0], color2=[1.0, 1.0, 1.0, 1.0]):
    nchannels=len(color1)
    img = np.zeros([img_width, img_width, nchannels], np.float32)
    img[:, :, 0:4] = color1
    nrows = img_width / patch_width
    for j in range(nrows):
        start_y = j * patch_width
        for i in range(nrows):
            start_x = i * patch_width * 2 + patch_width * (j % 2)
            img[start_x:start_x + patch_width, start_y:start_y + patch_width, 0:4] = color2
    return img

def colourfulness(img):
    """
    Returns colourfulness as defined in: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf

    args
    :img - RGB image, dtype is converted to float32 internally for safe usage
    """
    img = img.astype(np.float32)

    assert(img.shape[-1] >= 3), 'Image must have at least 3 channels!'

    rg = np.abs(img[..., 0] - img[..., 1]).flatten()
    yb = np.abs(0.5 *(img[..., 0] + img[..., 1]) - img[..., 2]).flatten()

    stdrg = np.std(rg)
    meanrg = np.mean(rg)

    stdyb = np.std(yb)
    meanyb = np.mean(yb)

    stdrgyb = (stdrg**2 + stdyb**2)**0.5
    meanrgyb = (meanrg**2 + meanyb**2)**0.5

    C = stdrgyb + 0.3*meanrgyb

    return C
