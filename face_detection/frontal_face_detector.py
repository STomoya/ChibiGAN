
import argparse
import glob
import os
import logging

import numpy as np
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


'''Helpers'''

def is_bigger(box_size: tuple, min_size):
    '''is the box bigger than min_size
    '''
    return box_size[0] >= min_size or box_size[1] >= min_size

def to4points(xyxy):
    '''2 corners to 4 corners
    '''
    return (
        (xyxy[0], xyxy[1]), (xyxy[0], xyxy[3]),
        (xyxy[2], xyxy[1]), (xyxy[2], xyxy[3]))

def any_inside(face, eye):
    '''is any of the corner inside the face box?
    '''
    eye_points = to4points(eye)
    for point in eye_points:
        if face[0] <= point[0] <= face[2] and face[1] <= point[1] <= face[3]:
            return True
    return False

def center(x1, x2):
    '''center of two points.
    '''
    return (x1 + x2) / 2

def clamp(x, size):
    '''clamp x to [0, size].
    '''
    return min(size, max(x, 0))

def calc_rotation_params(eyes):
    '''calc rotation degree and center from eyes.
    '''
    # eye box centers
    right_center = (center(eyes['right'][0], eyes['right'][2]), center(eyes['right'][1], eyes['right'][3]))
    left_center  = (center(eyes['left'][0], eyes['left'][2]), center(eyes['left'][1], eyes['left'][3]))
    # center of two eyes
    face_center = (center(right_center[0], left_center[0]), center(right_center[1], left_center[1]))
    # rotation
    radian = np.arctan((left_center[1] - right_center[1]) / - (left_center[0] - right_center[0] + 1e-10))
    rotation = - np.rad2deg(radian)

    return face_center, rotation

def save_params(path, line: str):
    '''save cropping parameters to a file.
    '''
    if not line.endswith('\n'):
        line = line + '\n'
    with open(path, 'a') as fp:
        fp.write(line)




class Detector:
    '''Frontal face detector for illustrated characters.

    Arguments:
        output: str
            The folder to save the cropped images.
        param_file: str (default: None)
            File to save the parameters used to crop the faces.
            NOTE: The file will always be opened with mode=='a'.
        weights: str (default: './weights/icartoonface_fasterrcnn_R101_FPN3x_FF.pth')
            Path to the weight files.
        threshold: float (default: 0.8)
            Threshold.
    '''
    def __init__(self,
        output: str, param_file: str=None,
        weights='./weights/icartoonface_fasterrcnn_R101_FPN3x_FF.pth',
        threshold: float = 0.8
    ) -> None:

        cfg = get_cfg()
        cfg.OUTPUT_DIR = output
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        self.config = cfg
        self.param_file = param_file
        if self.param_file is not None:
            save_params(self.param_file, 'path,box_x1,box_y1,box_x2,box_y2,center_x,center_y,rotation')

        self.threshold = threshold
        self.predictor = DefaultPredictor(cfg)
        if not os.path.exists(self.predictor.cfg.OUTPUT_DIR):
            os.makedirs(self.predictor.cfg.OUTPUT_DIR)

    def __call__(self, image):
        '''forward'''
        return self.predictor(image)

    def detect(self,
        path: str, min_size: int=None, verbose: bool=True
    ) -> None:
        '''Detect on an image.

        Arguments:
            path: str
                Path to the image.
            min_size: int (default: None)
                Minimum size of the bounding box to be saved.
            verbose: bool (default: False)
                Log simple results.
        '''

        # load file
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warn(f'Could not read {path}. Exit.')
            return
        height, width = image.shape[:2]

        if min_size is None:
            min_size = 0

        # predict
        instances = self(image)['instances']
        thres_instances = instances[instances.scores > self.threshold]
        faces      = thres_instances[thres_instances.pred_classes == 0].get('pred_boxes')
        right_eyes = thres_instances[thres_instances.pred_classes == 1].get('pred_boxes')
        left_eyes  = thres_instances[thres_instances.pred_classes == 2].get('pred_boxes')
        front_faces = self._find_front_faces(faces, right_eyes, left_eyes)

        # crop
        filename = os.path.splitext(os.path.basename(path))[0]
        count = 0
        for face in front_faces:
            face_image = image.copy()
            face_box = face['box']

            # adjust face box
            # expand
            box_height, box_width = face_box[3] - face_box[1], face_box[2] - face_box[0]
            face_box = (
                clamp(face_box[0]-box_width//2, width), clamp(face_box[1]-box_height, height),
                clamp(face_box[2]+box_width//2, width), clamp(face_box[3]+box_height//4, height)
            )
            box_height, box_width = face_box[3] - face_box[1], face_box[2] - face_box[0]
            box_center = (face_box[0] + box_width//2, face_box[1] + box_height//2)
            # adjust box to be nearly square
            box_size_square = (box_width + box_height) // 4
            face_box = list(map(int, [
                clamp(box_center[0]-box_size_square, width), clamp(box_center[1]-box_size_square, height),
                clamp(box_center[0]+box_size_square, width), clamp(box_center[1]+box_size_square, height)]))
            box_height, box_width = face_box[3] - face_box[1], face_box[2] - face_box[0]
            box_center = (face_box[0] + box_width//2, face_box[1] + box_height//2)

            if is_bigger((box_height, box_width), min_size):
                # rotate
                rotmat = cv2.getRotationMatrix2D(face['center'], face['rotation'], 1.)
                face_image = cv2.warpAffine(face_image, rotmat, (width, height), borderMode=cv2.BORDER_REPLICATE)

                # horizontal shift
                transmat = np.array([[1, 0, box_center[0]-face['center'][0]], [0, 1, 0]])
                face_image = cv2.warpAffine(face_image, transmat, (width, height), borderMode=cv2.BORDER_REPLICATE)

                # crop
                face_image = face_image[face_box[1]:face_box[3], face_box[0]: face_box[2], :]

                # save
                count += 1
                output_filename = os.path.join(self.config.OUTPUT_DIR, f'{filename}_{count}.jpg')
                cv2.imwrite(output_filename, face_image)

                # log
                if verbose:
                    message = '{src} -> {dst}, size {size}'.format(src=path, dst=output_filename, size=face_image.shape[:2])
                    logger.info(message)

                # save
                if self.param_file is not None:
                    face_box_str = ','.join(map(str, face_box))
                    center_str = ','.join(map(str, face['center']))
                    rotation_str = str(face['rotation'])
                    params = ','.join([path, face_box_str, center_str, rotation_str])
                    save_params(self.param_file, params)


    def _find_front_faces(self, faces, right_eyes, left_eyes):
        '''find faces that have two eyes, each from right and left, within bbox.'''

        front_faces = []
        for face in faces:
            face_xyxy = face.to(torch.long).tolist()
            eyes = {'right': None, 'left': None}

            for right_eye in right_eyes:
                right_eye_xyxy = right_eye.to(torch.long).tolist()
                if any_inside(face_xyxy, right_eye_xyxy):
                    eyes['right'] = right_eye_xyxy
                    break
            if eyes['right'] is None: # skip current face if no right eye
                continue

            for left_eye in left_eyes:
                left_eye_xyxy = left_eye.to(torch.long).tolist()
                if any_inside(face_xyxy, left_eye_xyxy):
                    eyes['left'] = left_eye_xyxy
                    break
            if eyes['left'] is None: # skip current face if no left eye
                continue

            face_center, rotation = calc_rotation_params(eyes)
            front_faces.append(dict(
                box=face_xyxy, center=face_center, rotation=rotation
            ))

        return front_faces



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input. Either a file or folder of image file.')
    parser.add_argument('weights', help='weights of the Faster R-CNN.')
    parser.add_argument('--threshold', default=0.8, type=float, help='Threshold')
    parser.add_argument('--min-size', type=int, help='Minimum size of the cropped images.')
    parser.add_argument('--output', default='./output', help='Folder to save cropped faces.')
    parser.add_argument('--param-file', help='If given, save parameters used to crop faces to this file')
    parser.add_argument('--quiet', default=False, action='store_true', help='No verbose.')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if os.path.isdir(args.input):
        paths = glob.glob(os.path.join(args.input, '*'))
    elif os.path.isfile(args.input):
        paths = [args.input]
    else:
        raise Exception(f'Something is wrong with the option "input"')

    model = Detector(args.output, args.param_file, args.weights, args.threshold)
    for path in paths:
        model.detect(path, args.min_size, not args.quiet)

if __name__=='__main__':
    main()
