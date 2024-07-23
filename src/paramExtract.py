class paramExtract():
    
    def __init__(self, videoFile, resultFile, modelPath, target_subject_id=0):
        import joblib
        import torch
        import numpy as np
        from smplx import SMPL
        import cv2 as cv

        self.VIDEO_PATH = videoFile

        self.SMPL_MODEL_PTH = modelPath
        OUTPUT_FILE_PATH = resultFile


        wham_results = joblib.load(self.OUTPUT_FILE_PATH)[target_subject_id]

        pose_world = wham_results["pose_world"]
        trans_world = wham_results["trans_world"]
        betas = wham_results["betas"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        body_model = SMPL(self.SMPL_MODEL_PTH, gender='neutral', num_betas=10).to(device)

        toTensor = lambda x: torch.from_numpy(x).float().to(device)

        smpl_kwargs = dict(
            global_orient=toTensor(pose_world[..., :3]),
            body_pose=toTensor(pose_world[..., 3:]),
            betas=toTensor(betas),
            transl=toTensor(trans_world)
        )

        smpl_output = body_model(**smpl_kwargs)

        joints = smpl_output.joints
        vertices = smpl_output.vertices

        self.pose_world = joints

        self.stepWidthLeft = None
        self.stepWidthRight = None
        self.stepLengthLeft = None
        self.stepLengthRight = None
        self.cadence = None

    def __init__(self, resultFile, modelPath, target_subject_id=0):
        import joblib
        import torch
        import numpy as np
        from smplx import SMPL
        import cv2 as cv

        self.videoFile = None

        self.SMPL_MODEL_PTH = modelPath
        self.OUTPUT_FILE_PATH = resultFile


        wham_results = joblib.load(self.OUTPUT_FILE_PATH)[target_subject_id]

        pose_world = wham_results["pose_world"]
        trans_world = wham_results["trans_world"]
        betas = wham_results["betas"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        body_model = SMPL(self.SMPL_MODEL_PTH, gender='neutral', num_betas=10).to(device)

        toTensor = lambda x: torch.from_numpy(x).float().to(device)

        smpl_kwargs = dict(
            global_orient=toTensor(pose_world[..., :3]),
            body_pose=toTensor(pose_world[..., 3:]),
            betas=toTensor(betas),
            transl=toTensor(trans_world)
        )

        smpl_output = body_model(**smpl_kwargs)

        joints = smpl_output.joints
        vertices = smpl_output.vertices

        self.pose_world = joints

        self.stepWidthLeft = None
        self.stepWidthRight = None
        self.stepLengthLeft = None
        self.stepLengthRight = None
        self.cadence = None

    
    def identifyFootfall(self): 
        frameIDsLeft = [[0]]
        frameIDsRight = [[0]]

        leftIndex = 7
        rightIndex = 8


        frameCounter = 0
        leftMin = 999
        leftMax = -999

        rightMin = 999
        rightMax = -999

        groundedLeft = True
        groundedRight = True

        lastUpLeft = 999
        lastUpRight = 999

        lastFourLeft = []
        lastFourRight = []

        thresholdUp = 0.035
        thresholdDown = 0.01

        counter = 0
        for frame in self.pose_world:
            #extracting a coordinates for left and right ankle
            zPosLeft = frame[leftIndex][1]
            zPosRight = frame[rightIndex][1]
            #Checking how many frames have passed
            if(len(lastFourLeft) == 4):

                
                #left footfall
                if(groundedLeft):
                    if (zPosLeft - lastFourLeft[3] > thresholdUp):
                        groundedLeft = False
                        lastUpLeft = lastFourLeft[3]
                        frameIDsLeft[len(frameIDsLeft) - 1].append(counter - 3)
                
                else:
                    if (abs(zPosLeft - lastFourLeft[3]) < thresholdDown and zPosLeft - lastUpLeft < 0.005):
                        groundedLeft = True
                        frameIDsLeft.append([counter - 3])


                #right footfall
                if(groundedRight):
                    if (zPosRight - lastFourRight[3] > thresholdUp):
                        groundedRight = False
                        lastUpRight = lastFourRight[3]
                        frameIDsRight[len(frameIDsRight) - 1].append(counter - 2)

                else:
                    if (abs(zPosRight - lastFourRight[3]) < thresholdDown and zPosRight - lastUpRight < 0.005):
                        groundedRight = True
                        frameIDsRight.append([counter - 2])


                    
                lastFourLeft[3] = lastFourLeft[2]
                lastFourLeft[2] = lastFourLeft[1]
                lastFourLeft[1] = lastFourLeft[0]
                lastFourLeft[0] = zPosLeft

                lastFourRight[3] = lastFourRight[2]
                lastFourRight[2] = lastFourRight[1]
                lastFourRight[1] = lastFourRight[0]
                lastFourRight[0] = zPosRight

            elif(len(lastFourLeft) == 3):
                lastFourLeft.append(lastFourLeft[2])
                lastFourLeft[2] = lastFourLeft[1]
                lastFourLeft[1] = lastFourLeft[0]
                lastFourLeft[0] = zPosLeft

                lastFourRight.append(lastFourRight[2])
                lastFourRight[2] = lastFourRight[1]
                lastFourRight[1] = lastFourRight[0]
                lastFourRight[0] = zPosRight



            elif(len(lastFourLeft) == 2):
                lastFourLeft.append(lastFourLeft[1])
                lastFourLeft[1] = lastFourLeft[0]
                lastFourLeft[0] = zPosLeft

                lastFourRight.append(lastFourRight[1])
                lastFourRight[1] = lastFourRight[0]
                lastFourRight[0] = zPosRight

            elif(len(lastFourLeft) == 1):
                lastFourLeft.append(lastFourLeft[0])
                lastFourLeft[0] = zPosLeft

                lastFourRight.append(lastFourRight[0])
                lastFourRight[0] = zPosRight

            else:
                lastFourLeft.append(zPosLeft)
                lastFourRight.append(zPosRight)


                
                
            counter += 1

            
        
        return [frameIDsLeft, frameIDsRight]
    

    #Extract Step Width for each foot
    def calculateStepWidth(self):
        if (self.stepWidthLeft != None):
            return [self.stepWidthLeft, self.stepWidthRight]
        import math
        
        #function identifyFootfall returns arrays that record the footfall 
        [frameIDsLeft, frameIDsRight] = self.identifyFootfall()

        #Checking if sufficient data was collected
        if (len(frameIDsLeft) <= 1 or len(frameIDsRight) <= 1):
            raise Exception("Error: Insufficient data collected")

        frameDiff = abs(len(frameIDsLeft) - len(frameIDsRight))
        misMatch = None
        

        #Determining which foot had more footfalls
        if frameDiff == 0:
            misMatch = False
            minFoot = frameIDsLeft
            print("Footfalls even")
        elif frameDiff == 1:
            misMatch = True
            minFoot = frameIDsLeft if len(frameIDsLeft) < len(frameIDsRight) else frameIDsRight
            print("Footfalls even")
        else:
            raise Exception(f"Error: Footfall MisMatch in Data by {frameDiff} keypoints")
            
        #firstFoot stores the foot data of the foot with the first footfall 
        firstFoot = frameIDsLeft if frameIDsLeft[0][1] < frameIDsRight[0][1] else frameIDsRight
        #nextFoot stores the other foot
        nextFoot = frameIDsLeft if firstFoot == frameIDsRight else frameIDsRight

        #Assigning respective keypoint indices
        firstIndex = 7 if firstFoot == frameIDsLeft else 8
        nextIndex = 7 if nextFoot == frameIDsLeft else 8


        #index for pelvic keypoint
        pelvicIndex = 0


        widthsFirst = []
        widthsNext = []

        #Returns function params given 2 x, y coordinates
        def slope_intercept(x_1, y_1, x_2, y_2):
            slope = (y_2 - y_1)/(x_2 - x_1)
            y_intercept = y_1 - slope * x_1

            return [slope, y_intercept]
        
        # Returns function params given slope and point
        def point_slope(slope, x, y):
            y_intercept = y - slope * x

            return [slope, y_intercept]
        
        #returns intercept given two lines
        def intercept(slope1, intercept1, slope2, intercept2):
            x = (intercept2 - intercept1) / (slope1 - slope2)
            y = slope1 * x + intercept1
            return[x, y]
        
        #dist function
        dist = lambda x1, y1, x2, y2: math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

        for i in range(0, len(minFoot)):

            if (i == 0):
            
                dataFirst1 = self.pose_world[firstFoot[i][1]][firstIndex]
                dataNext = self.pose_world[nextFoot[i][1]][nextIndex]
                dataFirst2 = self.pose_world[firstFoot[i + 1][1]][firstIndex]
                [firstLineSlope, firstLineIntercept] = slope_intercept(dataFirst1[0], dataFirst1[2], dataFirst2[0], dataFirst2[2])

                perp_slope = 1 / firstLineSlope

                [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext[0], dataNext[2])

                [nextInterceptX, nextInterceptY] = intercept(firstLineSlope, firstLineIntercept, nextFootSlope, nextFootIntercept)

                widthsNext.append(dist(dataNext[0], dataNext[2], nextInterceptX, nextInterceptY))
                
            
            elif (i < len(minFoot) - 1 and i > 0):

                dataFirst1 = self.pose_world[firstFoot[i][0]][firstIndex]
                dataNext1 = self.pose_world[nextFoot[i - 1][0]][nextIndex]
                dataFirst2 = self.pose_world[firstFoot[i + 1][0]][firstIndex]
                dataNext2 = self.pose_world[nextFoot[i][0]][nextIndex]

                [firstLineSlope, firstLineIntercept] = slope_intercept(dataFirst1[0], dataFirst1[2], dataFirst2[0], dataFirst2[2])
                [nextLineSlope, nextLineIntercept] = slope_intercept(dataNext1[0], dataNext1[2], dataNext2[0], dataNext2[2])

                first_perp_slope = 1 / firstLineSlope
                next_perp_slope = 1 / nextLineSlope

                [firstFootSlope, firstFootIntercept] = point_slope(next_perp_slope, dataFirst1[0], dataFirst1[2])
                [nextFootSlope, nextFootIntercept] = point_slope(first_perp_slope, dataNext2[0], dataNext2[2])

                [firstInterceptX, firstInterceptY] = intercept(nextLineSlope, nextLineIntercept, firstFootSlope, firstFootIntercept)
                [nextInterceptX, nextInterceptY] = intercept(firstLineSlope, firstLineIntercept, nextFootSlope, nextFootIntercept)

                widthsFirst.append(dist(dataFirst1[0], dataFirst1[2], firstInterceptX, firstInterceptY))
                widthsNext.append(dist(dataNext2[0], dataNext2[2], nextInterceptX, nextInterceptY))

            else:
                dataFirst = self.pose_world[firstFoot[i][0]][firstIndex]
                dataNext1 = self.pose_world[nextFoot[i - 1][1]][nextIndex]
                dataNext2 = self.pose_world[nextFoot[i][0]][nextIndex]

                [nextLineSlope, nextLineIntercept] = slope_intercept(dataNext1[0], dataNext1[2], dataNext2[0], dataNext2[2])

                next_perp_slope = 1 / nextLineSlope

                [firstFootSlope, firstFootIntercept] = point_slope(next_perp_slope, dataFirst[0], dataFirst[2])

                [firstInterceptX, firstInterceptY] = intercept(nextLineSlope, nextLineIntercept, firstFootSlope, firstFootIntercept)

                widthsFirst.append(dist(dataFirst[0], dataFirst[2], firstInterceptX, firstInterceptY))

                

        

            

            


     
        #If there is a mis-match in footfall count, we will calculate an extra step width for the start foot. 
        #Note that it cannot be the case that there are more footsteps of the next foot, since this would imply that 
        #the next foot has 2 or more identified footfalls than the first foot
        if (misMatch):
            dataLast2 = self.pose_world[firstFoot[len(minFoot)][0]][firstIndex]
            dataLast1 = self.pose_world[firstFoot[len(minFoot) - 1][0]][firstIndex]

            dataNext = self.pose_world[nextFoot[len(minFoot) - 1][0]][nextIndex]
            

            [lastLineSlope, lastLineIntercept] = slope_intercept(dataLast1[0], dataLast1[2], dataLast2[0], dataLast2[2])

            perp_slope = 1 / lastLineSlope

            [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext[0], dataNext[2])

            [nextInterceptX, nextInterceptY] = intercept(lastLineSlope, lastLineIntercept, nextFootSlope, nextFootIntercept)

            widthsNext.append(dist(dataNext[0], dataNext[2], nextInterceptX, nextInterceptY))

        #Calculating averages and returning left and right step widths

        totalValFirst = 0
        for val in widthsFirst:
            totalValFirst += val

        totalValNext = 0
        for val in widthsNext:
            totalValNext += val



        if (firstFoot == frameIDsLeft):
            self.stepWidthLeft = totalValFirst/len(widthsFirst)
            self.stepWidthRight = totalValNext/len(widthsNext)
            return [self.stepWidthLeft, self.stepWidthRight]
        else:
            self.stepWidthLeft = totalValNext/len(widthsNext)
            self.stepWidthRight = totalValFirst/len(widthsFirst)
            return [self.stepWidthLeft, self.stepWidthRight]
    

        
    
    #Extract Step Length for each foot
    def calculateStepLength(self):
        import math

        if (self.stepLengthLeft != None):
            return [self.stepLengthLeft, self.stepLengthRight]
        
        #function identifyFootfall returns arrays that record the footfall 
        [frameIDsLeft, frameIDsRight] = self.identifyFootfall()

        if (len(frameIDsLeft) <= 1 or len(frameIDsRight) <= 1):
            raise Exception("Error: Insufficient data collected")

        frameDiff = abs(len(frameIDsLeft) - len(frameIDsRight))
        misMatch = None

        #Determining which foot had more footfalls
        if frameDiff == 0:
            misMatch = False
            minFoot = frameIDsLeft if len(frameIDsLeft) < len(frameIDsRight) else frameIDsRight
            print("Footfalls even")
        elif frameDiff == 1:
            misMatch = True
            minFoot = frameIDsLeft if len(frameIDsLeft) < len(frameIDsRight) else frameIDsRight
            print("Footfalls even")
        else:
            raise Exception(f"Error: Footfall MisMatch in Data by {frameDiff} keypoints")

            
        #firstFoot stores the foot data of the foot with the first footfall 
        firstFoot = frameIDsLeft if frameIDsLeft[0][1] < frameIDsRight[0][1] else frameIDsRight
        #nextFoot stores the other foot
        nextFoot = frameIDsLeft if firstFoot == frameIDsRight else frameIDsRight

        #Assigning respective keypoint indices
        firstIndex = 7 if firstFoot == frameIDsLeft else 8
        nextIndex = 7 if nextFoot == frameIDsLeft else 8


        #index for pelvic keypoint
        pelvicIndex = 0


        lengthsFirst = []
        lengthsNext = []

        #Returns function params given 2 x, y coordinates
        def slope_intercept(x_1, y_1, x_2, y_2):
            slope = (y_2 - y_1)/(x_2 - x_1)
            y_intercept = y_1 - slope * x_1

            return [slope, y_intercept]
        
        # Returns function params given slope and point
        def point_slope(slope, x, y):
            y_intercept = y - slope * x

            return [slope, y_intercept]
        
        #returns intercept given two lines
        def intercept(slope1, intercept1, slope2, intercept2):
            x = (intercept2 - intercept1) / (slope1 - slope2)
            y = slope1 * x + intercept1
            return[x, y]
        
        #dist function
        dist = lambda x1, y1, x2, y2: math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

        #Iterating through steps and Calculating step Length for each successive two steps

        for i in range(0, len(minFoot)):
            if (i == 0):
                dataNext1 = self.pose_world[nextFoot[i][1]][nextIndex]
                dataFirst2 = self.pose_world[firstFoot[i + 1][0]][firstIndex]

                data_pelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                data_pelvic2 = self.pose_world[firstFoot[i + 1][0]][pelvicIndex]

                [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[2], data_pelvic2[0], data_pelvic2[2])

                perp_slope = 1 / plineSlope

                [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext1[0], dataNext1[2])
                [firstFootSlope2, firstFootIntercept2] = point_slope(perp_slope, dataFirst2[0], dataFirst2[2])

                [nextInterceptX, nextInterceptY] = intercept(plineSlope, plineIntercept, nextFootSlope, nextFootIntercept)
                [firstInterceptX2, firstInterceptY2] = intercept(plineSlope, plineIntercept, firstFootSlope2, firstFootIntercept2)

                lengthsFirst.append(dist(nextInterceptX, nextInterceptY, firstInterceptX2, firstInterceptY2))

            elif (i < len(minFoot) - 1 and i > 0):
                dataFirst1 = self.pose_world[firstFoot[i][1]][firstIndex]
                dataNext1 = self.pose_world[nextFoot[i][1]][nextIndex]
                dataFirst2 = self.pose_world[firstFoot[i + 1][0]][firstIndex]

                data_pelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                data_pelvic2 = self.pose_world[firstFoot[i + 1][0]][pelvicIndex]

                [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[2], data_pelvic2[0], data_pelvic2[2])

                perp_slope = 1 / plineSlope

                [firstFootSlope1, firstFootIntercept1] = point_slope(perp_slope, dataFirst1[0], dataFirst1[2])
                [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext1[0], dataNext1[2])
                [firstFootSlope2, firstFootIntercept2] = point_slope(perp_slope, dataFirst2[0], dataFirst2[2])

                [firstInterceptX1, firstInterceptY1] = intercept(plineSlope, plineIntercept, firstFootSlope1, firstFootIntercept1)
                [nextInterceptX, nextInterceptY] = intercept(plineSlope, plineIntercept, nextFootSlope, nextFootIntercept)
                [firstInterceptX2, firstInterceptY2] = intercept(plineSlope, plineIntercept, firstFootSlope2, firstFootIntercept2)

                lengthsFirst.append(dist(nextInterceptX, nextInterceptY, firstInterceptX2, firstInterceptY2))
                lengthsNext.append(dist(firstInterceptX1, firstInterceptY1, nextInterceptX, nextInterceptY))

            else:
                dataFirst1 = self.pose_world[firstFoot[i][0]][firstIndex]
                dataNext1 = self.pose_world[nextFoot[i][0]][nextIndex]

                data_pelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                data_pelvic2 = self.pose_world[nextFoot[i][0]][pelvicIndex]

                [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[2], data_pelvic2[0], data_pelvic2[2])

                perp_slope = 1 / plineSlope

                [firstFootSlope1, firstFootIntercept1] = point_slope(perp_slope, dataFirst1[0], dataFirst1[2])
                [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext1[0], dataNext1[2])

                [firstInterceptX1, firstInterceptY1] = intercept(plineSlope, plineIntercept, firstFootSlope1, firstFootIntercept1)
                [nextInterceptX, nextInterceptY] = intercept(plineSlope, plineIntercept, nextFootSlope, nextFootIntercept)

                lengthsNext.append(dist(firstInterceptX1, firstInterceptY1, nextInterceptX, nextInterceptY))
    
        

        #If there is a mis-match in footfall count, we will calculate an extra step Length for the start foot. 
        #Note that it cannot be the case that there are more footsteps of the next foot, since this would imply that 
        #the next foot has 2 or more identified footfalls than the first foot
    
        if (misMatch):
            dataLast = self.pose_world[firstFoot[len(minFoot)][0]][firstIndex]
            dataPrior = self.pose_world[nextFoot[len(minFoot) - 1][0]][nextIndex]

            data_pelvic1 = self.pose_world[firstFoot[len(minFoot)][0]][firstIndex]
            data_pelvic2 = self.pose_world[firstFoot[len(minFoot) - 1][1]][firstIndex]

            [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[2], data_pelvic2[0], data_pelvic2[2])

            perp_slope = 1 / plineSlope

            [lastFootSlope, lastFootIntercept] = point_slope(perp_slope, dataLast[0], dataLast[2])
            [priorFootSlope, priorFootIntercept] = point_slope(perp_slope, dataPrior[0], dataPrior[2])

            [lastInterceptX, lastInterceptY] = intercept(plineSlope, plineIntercept, lastFootSlope, lastFootIntercept)
            [priorInterceptX, priorInterceptY] = intercept(plineSlope, plineIntercept, priorFootSlope, priorFootIntercept)

            lengthsFirst.append(dist(priorInterceptX, priorInterceptY, lastInterceptX, lastInterceptY))

        totalValFirst = 0
        for val in lengthsFirst:
            totalValFirst += val

        totalValNext = 0
        for val in lengthsNext:
            totalValNext += val

        if (firstFoot == frameIDsLeft):
            self.stepLengthLeft = totalValFirst/len(lengthsFirst)
            self.stepLengthRight = totalValNext/len(lengthsNext)
            return [self.stepLengthLeft, self.stepLengthRight]
        else:
            self.stepLengthLeft = totalValNext/len(lengthsNext)
            self.stepLengthRight = totalValFirst/len(lengthsFirst)
            return [self.stepLengthLeft, self.stepLengthRight]

    #Calculates step cadence
    def calculateCadence(self):
        import cv2 as cv
        if (self.cadence != None):
            return self.cadence

        if (self.VIDEO_PATH == None):
            raise Exception("Error: Video Path not specified. Vido Path is necessary for Cadence calculation")

        video = cv.VideoCapture(self.VIDEO_PATH)
        # Get the FPS using the CAP_PROP_FPS property
        fps = video.get(cv.CAP_PROP_FPS)
        video.release()

        [frameIDsLeft, frameIDsRight] = self.identifyFootfall()

        firstFrame = frameIDsLeft[0][1] if frameIDsLeft[0][1] < frameIDsRight[0][1] else frameIDsRight[0][1]
        lastFrame = frameIDsLeft[len(frameIDsLeft) - 1][0] if frameIDsLeft[len(frameIDsLeft) - 1][0] > frameIDsRight[len(frameIDsRight) - 1][0] else frameIDsRight[len(frameIDsRight) - 1][0]

        duration = (lastFrame - firstFrame + 1) / fps
        stepCount = len(frameIDsLeft) + len(frameIDsRight) - 2

        self.cadence = stepCount * (60/duration)
        return self.cadence
        
    

test = paramExtract('output/demo/Normal_Andrew1/wham_output.pkl', 'dataset/body_models/smpl')
print(test.identifyFootfall())
print(f'Step Length: {test.calculateStepLength()}')
print(f'Step Width: {test.calculateStepWidth()}')


# print('left:')
# for footfall in test.identifyFootfall()[0]:
#     print(test.pose_world[footfall[0]][7])

# print('Right:')
# for footfall in test.identifyFootfall()[1]:
#     print(test.pose_world[footfall[0]][8])


        




        
