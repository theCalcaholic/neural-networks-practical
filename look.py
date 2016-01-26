import re
import os
import argparse
import Tkinter as tk
import tkFont

class DynamicLabel(tk.Label):
    def __init__(self, *args, **kwargs):
        tk.Label.__init__(self, *args, **kwargs)

        # clone the font, so we can dynamically change
        # it to fit the label width
        font = self.cget("font")
        base_font = tkFont.nametofont(self.cget("font"))
        self.font = tkFont.Font()
        self.font.configure(**base_font.configure())
        self.configure(font=self.font)

        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        text = self.cget("text")

        # first, grow the font until the text is too big,
        size = self.font.actual("size")
        while size < event.width:
            size += 1
            self.font.configure(size=size)

        # ... then shrink it until it fits
        while size > 1 and self.font.measure(text) > event.width:
            size -= 1
            self.font.configure(size=size)


activities = []
weights    = []


# command line arguments
# zoom: integer, must not be 0
# path: string, image directory
# noactivities: use if you don't want to display activities
# noweights: use if you don't want to display weights
# example: python newlook.py --zoom=10 --noweights
parser = argparse.ArgumentParser()
parser.add_argument('--zoom', type=int, default=1)
parser.add_argument('--path', type=str, default= os.getcwd())
parser.add_argument('--noactivities', action='store_false', default=True)
parser.add_argument('--noweights', action='store_false', default=True)
# magnification factor for images
args           = vars(parser.parse_args())
zoom           = args['zoom']
directory      = args['path'] + "/"
display_act    = args['noactivities']
display_weight = args['noweights']


# this function is only to supress output printed to terminal when quitting; following: http://courses.cms.caltech.edu/lead/lectures/lecture12.pdf
def exit_python(event):
    '''Exit Python when the event 'event' occurs.'''
    quit() # no arguments to quit


# draw base GUI
root = tk.Tk()
root.bind("<Button-3>", exit_python) # right mouse exit
root.bind("q", exit_python) # q exit


# read files and sort them (activities, weights)
def readFiles():
    dirs = os.listdir(directory)

    #regex
    activity = re.compile("obs_[a-zA-Z]*_[0-9]{1}.pgm")
    weight   = re.compile("obs_[a-zA-Z]*_[0-9]{1}_[0-9]{1}.pgm")

    for d in dirs:
        if activity.match(d):
            activities.append(d)

        elif weight.match(d):
            weights.append(d)


# get metainformation from a file
def getFileEssentials (filename):
    fobj  = open(filename, "r")
    line1 = fobj.readline()
    line2 = fobj.readline()
    line3 = fobj.readline()

    fobj.close()

    dummy, filetrunk = filename.split("obs_")
    filetrunk, dummy = filetrunk.split(".pgm")

    dummy, highS = line2.split("highS:")
    highS, lowS  = highS.split("lowS:")
    highS        = round(float(highS.strip()), 2)
    lowS         = round(float(lowS.strip()), 2)
    image_width, image_height = line3.split(" ")
    image_width  = zoom * int(image_width)
    image_height = zoom * int(image_height)

    return filetrunk, image_width, image_height, str(highS), str(lowS)


    # draw GUI: sort activities and weights and draw them 
    # activities are sorted from top to bottom along the
    # given characters and from left to right along the numbers
    # weights are sorted along the numbers from left to right
def drawFilesonGUI():
    # counting the rows while drawing activities:
    rowcounter = 0
    colcounter = 0

    if display_act:
        # sorting things in activities:
        # regex: group_1: character, group_2: number
        activity = re.compile("obs_([a-zA-Z]*)_([0-9]{1}).pgm")

        for a in activities:

            # get metadata
            filetrunk, image_width, image_height, highS, lowS = getFileEssentials(directory + a)

            # using char and number matches to create position in layout:
            m   = activity.match(a)
            row = ord(m.group(1)) - 64 # A=65
            col = int(m.group(2)) + 1



            # creating frame for image and description
            activity_frame = tk.LabelFrame(root, bd=0)
            # set frame to grid layout
            activity_frame.grid(row=row, column=col)

            # preparing image
            foto = tk.PhotoImage(file = directory + a)
            foto = foto.zoom(zoom,zoom)

            # creating label for image
            activity_lab = DynamicLabel(activity_frame, image=foto)
            # this step is necessary to keep the reference:
            activity_lab.image = foto
            # add to labelframe
            activity_lab.pack()

            # add name, min and max
            name        = filetrunk
            values      = lowS + " " + "..." + " "+highS
            labeltext   = name + ": " + values
            name_label = tk.Label(activity_frame, text=labeltext)

            # add to labelframe
            name_label.pack()



            # setting new row
            if row > rowcounter:
                rowcounter = row


    if display_weight:
        # sorting things in weights
        # regex: group_1: higher layer, group_2: lower layer
        weight   = re.compile("obs_[a-zA-Z]*_([0-9]{1})_([0-9]{1}).pgm")

        for w in weights:
            # get meta information
            filetrunk, image_width, image_height, highS, lowS = getFileEssentials(directory + w)

            # using layer matches to create position in layout:
            m            = weight.match(w)
            higher_layer = int(m.group(1))
            lower_layer  = int(m.group(2))

            # weights are displayed below the activities sorted by lower layer...
            row = rowcounter + 1 + lower_layer
            # ... and on their higher layer
            col = higher_layer + 1

            # creating frame for image and description
            weight_frame = tk.LabelFrame(root, bd=0)
            # set frame to grid layout
            weight_frame.grid(row=row, column=col)


            # preparing image
            foto = tk.PhotoImage(file = directory + w)
            foto = foto.zoom(zoom,zoom)

            # the image is attached to a label
            weight_lab  = tk.Label(weight_frame, image=foto)
            # this step is necessary to keep the reference:
            weight_lab.image = foto
            # add to weightframe
            weight_lab.pack()


            # add name label
            name        = filetrunk
            values      = lowS + " " + "..." + " "+highS
            labeltext   = name + ": " + values
            name_label  = tk.Label(weight_frame, text=labeltext)

            # add to weightframe
            name_label.pack()

            # setting new row
            if col > colcounter:
                columncounter = col





# this function allows to refresh the GUI interactively
def refreshGUI(event):
    # clear lists
    activities[:] = []
    weights[:]    = []

    # clear GUI
    for child in root.winfo_children():
        child.destroy()

    readFiles()
    drawFilesonGUI()
    root.mainloop()

root.bind("<Button-1>", refreshGUI) #left mouse refresh
readFiles()
drawFilesonGUI()
root.mainloop()
