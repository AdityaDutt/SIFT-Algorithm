# SIFT-Algorithm
Using sift algorithm to track objects and observing the effect of each parameter on performance.

## Usage:
Call function	<i> FindSomeKeypoints </i> to run the code.

Inputs:
FName1- An image file of the form {Name1}.{ext}

FName2- An image file that is an optional argument

Params- A Dictionary of keyword-value pairs to be used in SIFT. Params can have from 0 to the total number of SIFT parameters elements.

Outputs:

If the input consists of 1 image:
+ the image with the keypoints overlaid on it should be
written to a file with the name
Keypoints{Name1}.{ext}
+ the Params should be written to a text file called
ParamsKeypoints{Name1}.txt
If the input consists of 2 images:
+ the image with the keypoints overlaid on it should be
written to 2 files with the name
Keypoints{Name1}.{ext} and Keypoints{Name2}.{ext}
+ an image that somehow depicts the (best) matches between
the images should be written to a file with the name
Keypoints{Name1Name2}Match.{ext}
+ the Params should be written to a text file called
ParamsKeypoints{Name1Name2}.txt
