#!/bin/bash

#########################################################
#                                                       #
# This script can take a group of images and blend them #
# together to create a single image.                    #
#                                                       #
#########################################################
#                                                       #
# Required package: sudo apt install enfuse             #
# References: https://wiki.panotools.org/Enfuse         #
#                                                       #
#########################################################


enfuse -o output.png *.png


