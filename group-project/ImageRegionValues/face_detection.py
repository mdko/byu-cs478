import sys
import argparse
import cv2
import cv2.cv as cv
import numpy as np

def load_img(img_path):
    #img_color = cv.LoadImage(img_path)
    #return img_color
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv.CV_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    return img_gray

# clockwise
def rotate_image(image, angle):
    #if angle == 90 or angle == 270:
    #    timg = cv.CreateImage((img.height,img.width), img.depth, img.channels)
    #    cv.Transpose(img,timg)
    #elif angle == 180:
    #    timg = cv.CloneImage(img)

    #if angle == 0:
    #    return timg

    #if angle == 90:
    #    flipMode = 1
    #elif angle == 180:
    #    flipMode = -1
    #elif angle == 270:
    #    flipMode = 0
    #cv.Flip(timg, timg, flipMode=flipMode)

    #return timg
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    return result

cascade_fn='/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_fn)

def detect(img, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20),
           flags=cv.CV_HAAR_SCALE_IMAGE):

    rects = cascade.detectMultiScale(img, scaleFactor=scaleFactor,
                                     minNeighbors=minNeighbors,
                                     minSize=minSize, flags=flags)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

class FaceDetection:

    def arff_attr(self):
        output = ""
        for direction in ['up', 'right', 'down', 'left']:
            output += "@attribute faces_detected_" + direction + " real\n"
        return output

    def values(self, infile):
        img = load_img(infile)

        output = ""

        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rot_img = rotate_image(img, angle);
            else:
                rot_img = img
            try:
                rects = detect(rot_img)
            except cv2.error:
                rects = []
            num_faces = len(rects)
            if angle != 0:
                output += ", "
            output += str(num_faces)

        return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output the number of faces detected at each rotation.')
    parser.add_argument('--img', nargs=1, type=str, help='The path to the image, if this is absent, the arff attribute declorations are outputted.')

    args = parser.parse_args()

    face_detection = FaceDetection()

    if args.img:
        infile = args.img[0]
        sys.stdout.write(face_detection.values(args.img[0]))
    else:
        sys.stdout.write(face_detection.arff_attr())
