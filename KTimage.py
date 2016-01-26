# To run, first create a directory for the files:
#    mkdir /tmp/coco/
# The weights will be written into "/tmp/coco/" as ".pnm" image files.

# To see these image files with max and min values from the comment displayed, use look.tcl as:
#    look.tcl a w 0 1
# This assumes that weights have names like obs_W_1_0.pgm (only V, W allowed),
# while activations must have names like obs_A_1.pgm (letters A-Z allowed).
# (Parameter "a" is to display activation files and "w" is for weight files,
#  the number is that of the neural area/layer, e.g. 0 and 1)


import numpy
import sys
PYTHONVERSION = sys.version[0] # Python version "2" or "3" -- global variable (string, not int!)

def importimage (filename):

    lowS, highS = 0.0, 1.0

    f = open(filename, 'r')

    # file key: only "P2" (greyscale as readable ascii) or "P5" (greyscale as machine code characters)
    format = f.readline()[0:2]
    print ("opening {}  image format is {}".format(filename,format))

    # next line
    line = f.readline()

    # possible comment in file: maximum and minimum values to interprete the data values
    if  line[0] == "#":
        if  "highS:" in line and "lowS:" in line:
            indexhighS = line.find("highS:") + len("highS:")
            indexlowS = line.find("lowS:") + len("lowS:")
            highS = float(line[indexhighS:].split()[0])
            lowS = float(line[indexlowS:].split()[0])
        else:
            highS = 255.0
            lowS = 0.0
        print ("highS={} lowS={}".format(highS,lowS))

        # next line
        line = f.readline()

    # get the sizes of the image
    parts = line.split()
    width, height = int(parts[0]), int(parts[1])
    print ("width={} height={}".format(width,height))

    # next line (says the maximum value that the data are scaled to, i.e. the brightness value that is to be displayed as white)
    line = f.readline()
    maxchar = int(line)
    if maxchar != 255 and maxchar != 1:
        print ("warning: maxchar is not 255")
    print ("maxchar={}  now reading data".format(maxchar))

    # now read the data
    data = f.read()
    f.close()

    # write the data into a data vector of length height times width (i.e. the image geometry doesn't matter)
    values = numpy.zeros(height*width)
    if  format == "P5":
        if len(data) != height*width:
            print ("len(data) does not fit height*width!")
        for i in range(len(data)):
            values[i] = float(ord(data[i])) / float(maxchar) * (highS - lowS) + lowS
    elif format == "P2":
        liste = data.split()
        if len(liste) != height*width:
            print ("len(data liste) does not fit height*width!")
        for i in range(len(liste)):
            values[i] = float(int(liste[i])) / float(maxchar) * (highS - lowS) + lowS
    else:
        print ("format {} not recognized!".format(format))

    return values, height, width



def exportinfo (filename, highS, lowS):
    """used by exporttiles()
       inserts into the exported file a comment which looks e.g. like this:  # highS: 0.099849  lowS: -0.099849
       this can be used by the image viewer look.tcl to display max and min values
    """
    f = open(filename, 'rb')
    content = f.read()
    f.close()
    f = open(filename, 'wb')
    charcount = 0
    for char in content:
        f.write(char)
        if charcount == 2:
           f.write('# highS: %.6f  lowS: %.6f\n' % (highS, lowS))
        charcount += 1
    f.close()


# Writes a weight matrix X (or activity vector X) as a file.
# Each unit's input weight vector is regarded as a tile of size h*w,
# and also the output neuron layer is seen as a 2-dimensional sheet of size x*y.
def exporttiles (X, h, w, filename, x=None, y=None):
    """If all arguments given:
         displays a weight matrix X
         of output-dimension x*y (number of rows)
         and input-dimension h*w (number of columns)
         by writing into filename as a pgm file.
       If last two arguments missing:
         displays activation vector X
         with dimension h*w (height, width)
         by writing into filename as a pgm file.
    """

    if x is None:
        X = numpy.reshape(X, (1, h*w)) # blow the vector up to a matrix with one row
        x, y, = 1, 1
        frame = 0
    else:
        frame = 1 # show one pixel row around each neuron's input weight vector

    xy, hw = numpy.shape(X)
    if  (xy != x*y) or (hw != h*w):
        print ('imagetiles: size error')

    Y = numpy.zeros((frame + x*(h+frame), frame + y*(w+frame)))

    image_id = 0
    for xx in range(x):
        for yy in range(y):
            if image_id >= xy: 
                break
            tile = numpy.reshape (X[image_id], (h, w))
            beginH, beginW = frame + xx*(h+frame), frame + yy*(w+frame)
            Y[beginH : beginH+h, beginW : beginW+w] = tile
            image_id += 1

    # out-commented so not to use pylab, which makes problems on Windows -- note that PIL makes problems on Mac
    #im = Image.new ("L", (frame + y*(w+frame), frame + x*(h+frame)))
    #im.info = 'writing this comment into the file does not work!' # hence, exportinfo function necessary
    #im.putdata (Y.reshape((frame + x*(h+frame)) * (frame + y*(w+frame))), offset=-Y.min()*255.0/(Y.max()-Y.min()), scale=255.0/(Y.max()-Y.min()) )
    #im.save(filename, cmap=pylab.cm.jet)  # seems to ignore the colormap
    #exportinfo (filename,  numpy.max(X), numpy.min(X))

    f = open(filename, 'w')
    # write the header
    f.write("P5\n")
    f.write('# highS: %.6f  lowS: %.6f\n' % (numpy.max(X), numpy.min(X)))
    width, height = frame + y*(w+frame), frame + x*(h+frame)
    line = str(width) + " " + str(height) + "\n"
    f.write(line)
    f.write("255\n") # values range from 0 to 255
    f.close()
    # write the data
    if  numpy.max(Y) != numpy.min(Y):
       factor = 255.0 / (numpy.max(Y) - numpy.min(Y))
    else:
       factor = 0.0
    Y = (Y - numpy.min(Y)) * factor
    c = ''
    for i in range(height):
        for j in range(width):
            c += chr(int(Y[i][j]))
    f = open(filename, 'r+b')
    f.seek(0,2)
    if PYTHONVERSION == "2":
        f.write(c)
    else:
        f.write(c.encode("ISO-8859-1"))
    f.close()

