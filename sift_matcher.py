"""

Created on Sat Oct 19, 2019


@author: ADUTT

"""



import numpy as np

import cv2, os, itertools, math, sys

import matplotlib.pyplot as plt





# Plot images in 3 x 3 grid

def PlotgridIm(ImageList, xAxis, yAxis, xLabel, yLabel, allParams):

        

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 10),dpi=250,

                            subplot_kw={'xticklabels': [], 'yticks': []})

    ind = 0

    for ax in axs.flat:

        Im = ImageList[ind]

        Im = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)

        ax.imshow(Im)



        if ind % 3 == 0:            

            ax.set_ylabel(yAxis[math.floor(ind/3)])

        if ind > 5:

            ax.set_xlabel(xAxis[ind-6])

        ind += 1

      

    plt.tight_layout()

    fig.text(0.5, 0.01, xLabel, ha='center')

    fig.text(0.005, 0.5, yLabel, va='center', rotation='vertical')

    fig.text(0.5, 0.98, allParams, ha = 'center', va = 'center')



    return fig

            

            



# Read parameters from dictionary. If a parameter is missing, use default value in its place 

def ReadParams(Params, ParamNames) :

    

    Keys = [k for k in Params]

    defaultParams = [0, 3, .04, 10, 1.6] 

    newParams = []

    for i in range(len(ParamNames)) :

        if ParamNames[i] in Keys :

            newParams.append(Params.get(ParamNames[i]))

        else :

            newParams.append(defaultParams[i])



    return newParams







# Find keypoints in an image using sift 

def FindSomeKeypoints(FName1, Params, FName2 = "") :


    # FName2 default value is Null. For this code, there is no FName2 so no argument will be passed.

    OrgParamNames = ["nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold", "sigma"]

    

    im = cv2.imread(FName1) # By default cv2.imread reads all images as 8 bit images. So, 4 channel is automatically converted to 3 channel image.

    # Check if image is 4 channel image. (Not useful here)

    

    img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)



    # equalize the histogram of the Y channel

    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    

    # convert the YUV image back to RGB format

    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    

    channels = im.shape

    if channels[-1] == 4 :

        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)



    imName = (FName1.split("/"))[-1] # Get name of file

    imName, File1Extension = imName.split(".")



    # Read parametes passed as arguments

    InputParams = ReadParams(Params, OrgParamNames)



    # Apply sift using given parameters

    sift = cv2.xfeatures2d.SIFT_create(InputParams[0], int(InputParams[1]), InputParams[2], int(InputParams[3]), InputParams[4])        

    keypoints_sift, descriptors = sift.detectAndCompute(im,None)        

    keyIm = cv2.drawKeypoints(im, keypoints_sift, None)    



    # Write keypoints to a txt file

    cv2.imwrite("Keypoints"+imName+"."+File1Extension, keyIm)    



    if FName2 == "" :

        paramFile = open("ParamsKeypoints"+imName+".txt", 'w')

    else :

        imName2 = (FName2.split("/"))[-1] # Get name of file

        imName2, File2Extension = imName2.split(".")

        paramFile = open("ParamsKeypoints"+imName+imName2+".txt", 'w')



    for i in range(len(OrgParamNames)) :

        paramFile.write(OrgParamNames[i]+ " : "+str(InputParams[i]) )

        if i < len(OrgParamNames)-1 :

            paramFile.write("\n")

    paramFile.close()



    # Display input and output images

    DispIm = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    plt.imshow(DispIm)

    plt.title("Input Image")

    plt.show()

    

    keyIm = cv2.cvtColor(keyIm, cv2.COLOR_BGR2RGB)

    plt.imshow(keyIm)

    plt.title("Output Image")

    plt.show()



    #---------------------------------------------------------------------#

    # Experiment with different values, plot images and record observation

    #----------------------------------------------------------------------#



    nOctaveLayers = [1,3,5]

    contrastThresh = [.04, .1, .2]

    edgeThresh = [30,60,90]

    sigma = [1.0, 1.6, 2.2]



    defaultParams = [ 3, .04, 10, 1.6]

    Parameters = [nOctaveLayers, contrastThresh, edgeThresh, sigma]

    ParamNames = ['Num Octave Layers', 'Contrast Threshold', 'Edge Threshold', 'Sigma']



    ParametersIdx = [0,1,2,3]

    plotPairsIdx = list(itertools.combinations(ParametersIdx,2))



    # Check if there is only one image in input

    if FName2 == "" :



            for i in range(len(plotPairsIdx)):



                # Get which 2 combination of parameter indices to plot

                xIdx, yIdx = plotPairsIdx[i]  

                XAxisParams = Parameters[xIdx]

                YAxisParams = Parameters[yIdx]

                plot9Im = []

                paramTitles = []



                # Loop over the values in that one parameter list

                for x in range(len(XAxisParams)):



                    # Creates a 3 lists of lists the parameter values

                    output = [[XAxisParams[x], y] for y in YAxisParams]



                    for n in range(len(output)) :



                        # Grab the parameters to test

                        p1, p2 = output[n]

                        temp = np.copy(defaultParams)

                        updatedParams = list(temp)

                        updatedParams[xIdx] = p1

                        updatedParams[yIdx] = p2



                        # Run cv2 given these parameters

                        sift = cv2.xfeatures2d.SIFT_create(0,int(updatedParams[0]), updatedParams[1], int(updatedParams[2]), updatedParams[3])        

                        keypoints_sift, descriptors = sift.detectAndCompute(im,None)        

                        keyIm = cv2.drawKeypoints(im, keypoints_sift, None)        

                        plot9Im.append(keyIm)

                        del(temp)



                # Plot the 9 images in a 3 x 3 grid



                title = [[ParamNames[j],defaultParams[j]] for j in ParametersIdx if j != xIdx and j != yIdx]

                #fig = PlotgridIm(plot9Im,YAxisParams,XAxisParams, ParamNames[yIdx], ParamNames[xIdx], title)

                #fig.savefig('Run'+str(i+1)+'.'+File1Extension)







    elif FName2 != "" : # If two images are passed.



            # Experiment with different values on both images and then match keypoints between those images using user passed arguments

            FileExtension = []

            FileExtension.append(File1Extension)

            FileExtension.append(File2Extension)



            descriptorsList = []

            keypointsList = []

            keypointsImages = []

            im2 = cv2.imread(FName2) # By default cv2.imread reads all images as 8 bit images. So, 4 channel is automatically converted to 3 channel image.

            #image equalization

            img_yuv = cv2.cvtColor(im2, cv2.COLOR_BGR2YUV)

            

            # equalize the histogram of the Y channel

            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

            

            # convert the YUV image back to RGB format

            im2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # Check if image is 4 channel image. (Not useful here)

            channels = im2.shape

            if channels[-1] == 4 :

                im2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2BGR)



            im1 = np.copy(im)

            imName1 = imName

            countIm = 1



            # Run loop twice, because there are 2 images

            while countIm <=2 :          

                if countIm == 1 :

                    im = np.copy(im1)

                    imName = imName1

                elif countIm == 2 :

                    im = np.copy(im2)

                    imName = imName2



                for i in range(len(plotPairsIdx)):



                    # Get which 2 combination of parameter indices to plot

                    xIdx, yIdx = plotPairsIdx[i]  

                    XAxisParams = Parameters[xIdx]

                    YAxisParams = Parameters[yIdx]

                    plot9Im = []

                    paramTitles = []



                    # Loop over the values in that one parameter list

                    for x in range(len(XAxisParams)):



                        # Creates a 3 lists of lists the parameter values

                        output = [[XAxisParams[x], y] for y in YAxisParams]



                        for n in range(len(output)) :



                            # Grab the parameters to test

                            p1, p2 = output[n]

                            temp = np.copy(defaultParams)

                            updatedParams = list(temp)

                            updatedParams[xIdx] = p1

                            updatedParams[yIdx] = p2



                            # Run cv2 given these parameters

                            sift = cv2.xfeatures2d.SIFT_create(0,int(updatedParams[0]), updatedParams[1], int(updatedParams[2]), updatedParams[3])        

                            keypoints_sift, descriptors = sift.detectAndCompute(im,None)        

                            keypointsList.append(keypoints_sift)

                            descriptorsList.append(descriptors)

                            keyIm = cv2.drawKeypoints(im, keypoints_sift, None)        

                            keypointsImages.append(keyIm)

                            plot9Im.append(keyIm)

                            del(temp)



                    # Plot the 9 images in a 3 x 3 grid



                    title = [[ParamNames[j],defaultParams[j]] for j in ParametersIdx if j != xIdx and j != yIdx]

                    #fig = PlotgridIm(plot9Im,YAxisParams,XAxisParams, ParamNames[yIdx], ParamNames[xIdx], title)

                    #fig.savefig('IMG'+str(countIm)+'Run'+str(i+1)+'.'+FileExtension[countIm-1])

                    plt.clf()

                    plt.cla()

                    plt.close()



                countIm += 1

            



            # -- Match keypoints in both images using input parameters -- #

        

            #     Apply sift using given parameters

            sift = cv2.xfeatures2d.SIFT_create(InputParams[0], int(InputParams[1]), InputParams[2], int(InputParams[3]), InputParams[4])        

            keypoints_sift1, descriptors1 = sift.detectAndCompute(im1,None)        

            keyIm1 = cv2.drawKeypoints(im1, keypoints_sift1, None)    

            keypoints_sift2, descriptors2 = sift.detectAndCompute(im2,None)        

            keyIm2 = cv2.drawKeypoints(im2, keypoints_sift2, None)    

            # Write keypoints of 2nd image to a txt file

            cv2.imwrite("Keypoints"+imName2+"."+File1Extension, keyIm2)    

        

            # Use BF Matcher to match keypoints

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        

            matches = bf.match(descriptors1,descriptors2)

            matches = sorted(matches, key = lambda x:x.distance)

        

        

            # Take only best 100( We can change it to make image ) matches

            matches = matches[:40]

        

            # Draw matches between keypoints

            MatchImg = cv2.drawMatches(im1, keypoints_sift1, im2, keypoints_sift2, matches, im2, flags=2)

        

            # Write keypoints match image

            cv2.imwrite("Keypoints"+imName1+imName2+"Match."+File1Extension, MatchImg)    

        

            '''

            # Display matches between keypoints

            plt.imshow(MatchImg)

            plt.show()

            '''