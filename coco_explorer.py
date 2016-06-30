#!/usr/bin/env python

# Infinite loop over random images from COCO_Text showing their annotations.
#
# Adapted from IPython Notebook demo in this repo.
#
# Invoke as:
# ./coco_explorer.py <path-to-MSCOCO> <path to COCO_Text.json>

# Std
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
import skimage.io as io
import sys

# COCO
import coco_text

def main():

    dataDir = sys.argv[1]
    dataType = "train2014"

    plt.switch_backend("TkAgg")
    pylab.rcParams['figure.figsize'] = (15.0, 10.0)

    ct = coco_text.COCO_Text(sys.argv[2])
    ct.info()

    # get all images containing at least one instance of legible text
    imgIds = ct.getImgIds(imgIds=ct.train, 
                                catIds=[('legibility','legible')])

    while True:
        # pick one at random
        img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

        I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
        print '/images/%s/%s'%(dataType,img['file_name'])
        plt.figure()
        annIds = ct.getAnnIds(imgIds=img['id'])
        anns = ct.loadAnns(annIds)
        ct.showAnns(anns)
        plt.imshow(I)
        plt.show()

if "__main__" == __name__:
    main()


