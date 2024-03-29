# SIFT-Algorithm
Demonstration of sift algorithm to track objects and observing the effect of each parameter on performance.

SIFT stands for scale-invariant feature transform (SIFT). It is a feature detection algorithm in computer vision to detect and describe local features in images. It was published by David Lowe in 1999. It has sevral applications like image stitching, 3D modeling, gesture recognition and video tracking. 

## Usage:

Example 1: 

```
FindSomeKeypoints(img1, img2)

Output:

```

<img align="center" alt="Python" src="/sift_matcher.png" />
</br>

Example 2: 

```
FindSomeKeypoints(img1)

Output: 

```

<img align="center" alt="Python" src="/sift_features.png" />
</br>


## Function Description

### FindSomeKeypoints)

FindSomeKeypoints(img1, img2)


<b>Inputs:</b>

FName1- An image file of the form {Name1}.{ext}

FName2- An image file that is an optional argument

Params- A Dictionary of keyword-value pairs to be used in SIFT. Params can have from 0 to the total number of SIFT parameters elements.

<b>Outputs:</b>

If the input consists of 1 image:

+ the image with the keypoints overlaid on it are written to a file with the name Keypoints{Name1}.{ext}

+ the Params are written to a text file called ParamsKeypoints{Name1}.txt

If the input consists of 2 images:
+ the image with the keypoints overlaid on it is written to 2 files with the name Keypoints{Name1}.{ext} and Keypoints{Name2}.{ext}

+ an image that depicts the (best) matches between the images are written to a file with the name Keypoints{Name1Name2}Match.{ext}

+ the Params are written to a text file called ParamsKeypoints{Name1Name2}.txt


## Authors

* **Aditya Dutt** - [[Website]](https://www.adityadutt.com) 
* **Nicholas Kroeger** - [[Website]](https://kroegern1.github.io/)



