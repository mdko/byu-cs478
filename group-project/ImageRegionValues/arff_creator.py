import os, sys
import argparse
#import image_region_values_border as irv
#import image_region_values as irv
import image_region_values_border_var_avg as iev
#import face_detection
from sets import Set

class FileNameAttribute:
    def __init__(self, input_folder_path):
        self.input_folder_path = input_folder_path

    def arff_attr(self):
        files = os.listdir(self.input_folder_path)
        filesWithPath = [os.path.join(self.input_folder_path, f) for f in files]
        filesStr = '{"' + '", "'.join(filesWithPath) + '"}'
        return "@attribute fileName " + filesStr + '\n'

    def values(self, imagePath):
        return '"' + imagePath + '"'

class ClassAtribute:
    def __init__(self, input_folder_path):
        self.classes = Set()
        files = os.listdir(input_folder_path)
        for imageName in files:
            self.classes.add(self.values(imageName))

    def values(self, imagePath):
        return '"' + imagePath[imagePath.rindex('-') + 1 : imagePath.rindex('.')] + '"'

    def arff_attr(self):
        return '@attribute class { ' + ', '.join(self.classes) + ' }\n'

numHorizontal = 1
numVertical = 1

def create_arff(input_folder_path, output_file_path):
    featureCreators = [
            #irv.RegionValues(numHorizontal, numVertical, 'rgb'),
            iev.RegionValues('rgb'),
            #face_detection.FaceDetection(),
            FileNameAttribute(input_folder_path),
            ClassAtribute(input_folder_path)]
    print("Outputting attributes")
    files = os.listdir(input_folder_path)
    outputContents = "@relation images\n"

    for featureCreator in featureCreators:
        outputContents += featureCreator.arff_attr()

    print("Outputting data")
    outputContents += "@data\n"
    for imagePath in files:
        print("Processing file: " + imagePath)
        for i, featureCreator in enumerate(featureCreators):
            if i != 0:
                outputContents += ", "
            outputContents += featureCreator.values(os.path.join(input_folder_path, imagePath))
        outputContents += "\n"

    print("Writing File")
    outputFile = open(output_file_path, "w")
    outputFile.write(outputContents)
    outputFile.close()
    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an arff file from a folder of images.')
    parser.add_argument('input_folder_path', type=str,
                       help='Path to the folder of images to convert.')
    parser.add_argument('output_file_path', type=str,
                       help='Path to the folder where to save the file.')

    args = parser.parse_args()

    create_arff(args.input_folder_path, args.output_file_path)
