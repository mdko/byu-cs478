import os, sys
import Image
import colorsys
import argparse

class RegionValues:

    def __init__(self, numRegionsHorizontal, numRegionsVertical, encoding=None):
        self.numRegionsHorizontal = numRegionsHorizontal
        self.numRegionsVertical = numRegionsVertical
        if encoding == None:
            encoding = 'hsv'
        self.encoding = encoding

    def values(self, infile, outfile=None):
        output = ""

        im = Image.open(infile)
        out = im.resize((self.numRegionsHorizontal, self.numRegionsVertical), Image.ANTIALIAS)
        if outfile != None:
            out.save(outfile, "JPEG")
        assert im.getbands() == ('R', 'G', 'B')
        for x in xrange(self.numRegionsHorizontal):
            for y in xrange(self.numRegionsVertical):
                r, g, b = out.getpixel((x, y))
                if self.encoding == 'rgb':
                    components = (r / 256.0, g / 256.0, b / 256.0)
                elif self.encoding == 'hsv':
                    components = colorsys.rgb_to_hsv(r / 256.0, g / 256.0, b / 256.0)
                elif self.encoding == 'prgbl':
                    total = float(r + g + b)
                    if total < 1:
                        components = (.333333, .33333, .33333, 0)
                    else:
                        components = (r / total, g / total, b / total, total / (256 * 3))
                else:
                    raise Exception("Unknown format")
                if x != 0 or y != 0:
                    output += ", "
                output += ', '.join([str(c) for c in components])
        return output


    def arff_attr(self):
        output = ""
        for x in xrange(self.numRegionsHorizontal):
            for y in xrange(self.numRegionsVertical):
                if self.encoding == 'rgb':
                    components = ['r', 'g', 'b']
                elif self.encoding == 'hsv':
                    components = ['h', 's', 'v']
                elif self.encoding == 'prgbl':
                    components = ['pr', 'pg', 'pb', 'l']
                else:
                    raise Exception("Unknown format")
                for value in components:
                    output += "@attribute reg" + str(x) + "," + str(y) + value + " real\n"
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output average HSV values for equal sized regions of a photo file.')
    parser.add_argument('horizontal', metavar='H', type=int,
                       help='The number of regions to divide the picture in horizontally')
    parser.add_argument('vertical', metavar='V', type=int,
                       help='The number of regions to divide the picture in vertically')
    parser.add_argument('--img', nargs=1, type=str, help='The path to the image, if this is absent, the arff attribute declorations are outputted.')
    parser.add_argument('--outfile', nargs=1, type=str, help='Optionally store the reduced image to the give location')
    parser.add_argument('--encoding', nargs=1, type=str, help='Which encoding? (rgb, hsv).')

    args = parser.parse_args()

    outfile = None
    if args.outfile != None:
        outfile = args.outfile[0]

    encoding = None
    if args.encoding != None:
        encoding = args.encoding[0]

    region_values = RegionValues(args.horizontal, args.vertical, encoding=encoding)

    if args.img:
        infile = args.img[0]
        sys.stdout.write(region_values.values(args.img[0], outfile=outfile))
    else:
        sys.stdout.write(region_values.arff_attr())
