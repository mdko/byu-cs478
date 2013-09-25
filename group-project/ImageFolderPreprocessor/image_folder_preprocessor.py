import os, sys
import Image
import colorsys
import argparse
import random

def convert_folder(input_folder_path, output_folder_path, shrink_images=False):
    files = os.listdir(input_folder_path)

    for infile in files:
        print "Processing image: " + infile
        im = Image.open(os.path.join(input_folder_path, infile))
        width, height = im.size
        if height > width:
            w = width
            width = height
            height = w
            was_vertical = True
            im = im.rotate(90)
        else:
            was_vertical = False
        if not is_aspect_ratio_in_bounds(width, height):
            print "Image is not within aspect ratio bounds. Skipping: " + infile
            continue


        if shrink_images and width > 1024:
            height = int((1024.0 / width) * height)
            width = 1024
            im = im.resize((width, height), Image.ANTIALIAS)

        filename = os.path.splitext(infile)[0]

        if was_vertical:
                output_direction = "left"
        else:
                output_direction = "up"
        #outfile = os.path.join(output_folder_path, filename + "-" + output_direction + ".jpg")
        #im.save(outfile, "JPEG")

        if random.randint(1,2) == 1:
            im = im.rotate(180)
            if was_vertical:
                output_direction = "right"
            else:
                output_direction = "down"
        outfile = os.path.join(output_folder_path, filename + "-" + output_direction + ".jpg")
        im.save(outfile, "JPEG")


def is_aspect_ratio_in_bounds(width, height):
    maxRatio = 1024/650.0
    minRatio = 1024/800.0
    ratio = width / float(height)
    if ratio > maxRatio:
        return False
    if ratio < minRatio:
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a folder of images into specified naming format.')
    parser.add_argument('input_folder_path', type=str,
                       help='Path to the folder of images to convert.')
    parser.add_argument('output_folder_path', type=str,
                       help='Path to the folder where to save the images.')
    parser.add_argument('--shrink', action='store_true', help="Shrink the images down to a maximum of 1024 pixels")

    args = parser.parse_args()

    convert_folder(args.input_folder_path, args.output_folder_path, shrink_images=args.shrink)
