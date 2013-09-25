import os, sys
import Image
import colorsys
import argparse
from numpy import var

class RegionValues:

    def __init__(self, encoding):
        if encoding == None:
            encoding = 'hsv'
        self.encoding = encoding

    def get_encoding_components(self):
        if self.encoding == 'rgb':
            components = ['r', 'g', 'b']
        elif self.encoding == 'hsv':
            components = ['h', 's', 'v']
        elif self.encoding == 'prgbl':
            components = ['pr', 'pg', 'pb', 'l']
        else:
            raise Exception("Unknown format")
        return components

    def arff_attr(self):
        output = ""
        for x in xrange(4):
            for value in self.get_encoding_components():
                for sum_metric in ['avg', 'var']:
                    output += "@attribute reg" + str(x) + value + "-" + sum_metric + " real\n"
        return output

    def values(self, infile):

        im = Image.open(infile)
        assert im.getbands() == ('R', 'G', 'B')

        def get_components(x, y):
            r, g, b = im.getpixel((x, y))
            if self.encoding == 'rgb':
                components = (r / 256.0, g / 256.0, b / 256.0)
            elif self.encoding == 'hsv':
                components = colorsys.rgb_to_hsv(r / 256.0, g / 256.0, b / 256.0)
            elif self.encoding == 'prgbl':
                total = float(r + g + b)
                if total < 1:
                    components = (.333333, .33333, .33333, 0)
                else:
                    components = (r / total, g / total, b / total, total / 256 * 3)
            else:
                raise Exception("Unknown format")
            return components

        def pixel_attrs_str(x, y):
            components = get_components(x,y)
            return ', '.join([str(c) for c in components])


        def add_components_to_different_lists(lists, components):
            for i, component in enumerate(components):
                lists[i].append(component)

        def avg(l):
            return sum(l)  / len(l)


        # the four lines (top, bottom...) and the r, g, b in each
        lines = []
        for i in xrange(4):
            components = []
            for j in xrange(len(self.get_encoding_components())):
                components.append([])
            lines.append(components)
        for x in xrange(im.size[0]):
            add_components_to_different_lists(lines[0], get_components(x, 0))
            add_components_to_different_lists(lines[1], get_components(x, im.size[1] - 1))
        for y in xrange(1, im.size[1] - 1):
            add_components_to_different_lists(lines[2], get_components(0, y))
            add_components_to_different_lists(lines[3], get_components(im.size[0] - 1, y))


        output = ""
        for l in xrange(4):
            for j in xrange(len(self.get_encoding_components())):
                if l != 0 or j != 0:
                    output += ", "
                output += str(avg(lines[l][j]))
                output += ", "
                output += str(var(lines[l][j]))


        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output edge statistical data (mean, var).')
    parser.add_argument('--img', nargs=1, type=str, help='The path to the image, if this is absent, the arff attribute declorations are outputted.')
    parser.add_argument('--encoding', nargs=1, type=str, help='Which encoding? (rgb, hsv).')

    args = parser.parse_args()
    encoding = None
    if args.encoding != None:
        encoding = args.encoding[0]

    region_values = RegionValues(encoding=encoding)

    if args.img:
        infile = args.img[0]
        sys.stdout.write(region_values.values(args.img[0]))
    else:
        sys.stdout.write(region_values.arff_attr())
