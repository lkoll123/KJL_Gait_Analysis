class paramExtract():
    
    def __init__(self, videoFile, resultFile, target_subject_id=0):
        import joblib
        import torch
        import numpy as np
        from smplx import SMPL
        import cv2 as cv

        SMPL_MODEL_PTH = 'dataset/body_models/smpl'
        OUTPUT_FILE_PATH= 'output/demo/Hemiplegic_Sriharsha1/wham_output.pkl'





        wham_results = joblib.load(OUTPUT_FILE_PATH)[target_subject_id]

        pose_world = wham_results["pose_world"]
        trans_world = wham_results["trans_world"]
        betas = wham_results["betas"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        body_model = SMPL(SMPL_MODEL_PTH, gender='neutral', num_betas=10).to(device)

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

        self.stepWidth = None
        self.stepLength = None
        self.cadence = None

    def __init__(self, resultFile, target_subject_id=0):
        import joblib
        import torch
        import numpy as np
        from smplx import SMPL
        import cv2 as cv

        SMPL_MODEL_PTH = 'dataset/body_models/smpl'
        OUTPUT_FILE_PATH= 'output/demo/Hemiplegic_Sriharsha1/wham_output.pkl'





        wham_results = joblib.load(OUTPUT_FILE_PATH)[target_subject_id]

        pose_world = wham_results["pose_world"]
        trans_world = wham_results["trans_world"]
        betas = wham_results["betas"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        body_model = SMPL(SMPL_MODEL_PTH, gender='neutral', num_betas=10).to(device)

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

        self.stepWidth = None
        self.stepLength = None
        self.cadence = None

    
    def identifyFootfall(self):
        frameIDsLeft = []
        frameIDsRight = []


        frameCounter = 0
        for frame in self.keypoints:

            frameCounter += 1

        
        return [frameIDsLeft, frameIDsRight]
    

    #Extract Step Width for each foot
    def stepWidth(self):
        if (self.stepWidth != None):
            return self.stepWidth
        import math
        
        #function identifyFootfall returns arrays that record the footfall 
        [frameIDsLeft, frameIDsRight] = self.identifyFootfall()

        frameDiff = abs(len(frameIDsLeft) - len(frameIDsRight))
        misMatch = None
        

        #Determining which foot had more footfalls
        match(frameDiff):
            case 0:
                misMatch = False
                print("Footfalls even")
            
            case 1: 
                misMatch = True
                #Will store the foot with the least amount of footfalls, in case that there is a mismatch
                minFoot = frameIDsLeft if len(frameIDsLeft) < len(frameIDsRight) else frameIDsRight
                
                print("Footfalls even")

            case _:
                print(f"Error: Footfall MisMatch in Data by {frameDiff} keypoints")
                return []
            
        #firstFoot stores the foot data of the foot with the first footfall 
        firstFoot = frameIDsLeft if frameIDsLeft[0][0] < frameIDsRight[0][0] else frameIDsRight
        #nextFoot stores the other foot
        nextFoot = frameIDsLeft if firstFoot == frameIDsRight else frameIDsRight

        #Assigning respective keypoint indices
        firstIndex = 10 if firstFoot == frameIDsLeft else 11
        nextIndex = 10 if nextFoot == frameIDsLeft else 11


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
            
            dataFirst = self.pose_world[firstFoot[i][0]][firstIndex]
            dataNext = self.pose_world[nextFoot[i][0]][nextIndex]
            
            if (i < len(minFoot) - 1):

                dataPelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                dataPelvic2 = self.pose_world[firstFoot[i + 1][0]][pelvicIndex]

            else:
                dataPelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                dataPelvic2 = self.pose_world[nextFoot[i][0]][pelvicIndex]

            [plineSlope, plineIntercept] = slope_intercept(dataPelvic1[0], dataPelvic1[1], dataPelvic2[0], dataPelvic2[1])

            perp_slope = 1 / plineSlope

            [firstFootSlope, firstFootIntercept] = point_slope(perp_slope, dataFirst[0], dataFirst[1])
            [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext[0], dataNext[1])

            [firstInterceptX, firstInterceptY] = intercept(plineSlope, plineIntercept, firstFootSlope, firstFootIntercept)
            [nextInterceptX, nextInterceptY] = intercept(plineSlope, plineIntercept, nextFootSlope, nextFootIntercept)

            widthsFirst[i] = dist(dataFirst[0], dataFirst[1], firstInterceptX, firstInterceptY)
            widthsNext[i] = dist(dataNext[0], dataNext[1], nextInterceptX, nextInterceptY)


     
        #If there is a mis-match in footfall count, we will calculate an extra step width for the start foot. 
        #Note that it cannot be the case that there are more footsteps of the next foot, since this would imply that 
        #the next foot has 2 or more identified footfalls than the first foot
        if (misMatch):
            dataLast = self.pose_world[firstFoot[len(minFoot)][0]][firstIndex]
            

            dataPelvic2 = self.pose_world[firstFoot[len(minFoot)][0]][pelvicIndex]
            dataPelvic1 = self.pose_world[firstFoot[len(minFoot - 1)][0]][pelvicIndex]

            [plineSlope, plineIntercept] = slope_intercept(dataPelvic1[0], dataPelvic1[1], dataPelvic2[0], dataPelvic2[1])

            perp_slope = 1 / plineSlope

            [lastFootSlope, lastFootIntercept] = point_slope(perp_slope, dataLast[0], dataLast[1])

            [lastInterceptX, lastInterceptY] = intercept(plineSlope, plineIntercept, lastFootSlope, lastFootIntercept)

            widthsFirst[len(widthsFirst)] = dist(dataLast[0], dataLast[1], lastInterceptX, lastInterceptY)

        #Calculating averages and returning left and right step widths

        totalValFirst = 0
        for val in widthsFirst:
            totalValFirst += val

        totalValNext = 0
        for val in widthsNext:
            totalValNext += val

        if (firstFoot == frameIDsLeft):
            return [totalValFirst/len(widthsFirst), totalValNext/len(widthsNext)]
        else:
            return [totalValNext/len(widthsNext), totalValFirst/len(widthsFirst)]
    

        
    
    #Extract Step Length for each foot
    def stepLength(self):
        import math

        if (self.stepLength != None):
            return self.stepLength
        
        #function identifyFootfall returns arrays that record the footfall 
        [frameIDsLeft, frameIDsRight] = self.identifyFootfall()

        frameDiff = abs(len(frameIDsLeft) - len(frameIDsRight))
        misMatch = None

        #Determining which foot had more footfalls
        match(frameDiff):
            case 0:
                misMatch = False
                print("Footfalls even")
            
            case 1: 
                misMatch = True
                #Will store the foot with the least amount of footfalls, in case that there is a mismatch
                minFoot = frameIDsLeft if len(frameIDsLeft) < len(frameIDsRight) else frameIDsRight
                print("Footfalls even")

            case _:
                print(f"Error: Footfall MisMatch in Data by {frameDiff} keypoints")
                return []
            
        #firstFoot stores the foot data of the foot with the first footfall 
        firstFoot = frameIDsLeft if frameIDsLeft[0][0] < frameIDsRight[0][0] else frameIDsRight
        #nextFoot stores the other foot
        nextFoot = frameIDsLeft if firstFoot == frameIDsRight else frameIDsRight

        #Assigning respective keypoint indices
        firstIndex = 10 if firstFoot == frameIDsLeft else 11
        nextIndex = 10 if nextFoot == frameIDsLeft else 11


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


        for i in range(0, len(minFoot)):
            if (i < len(minFoot) - 1):
                dataFirst1 = self.pose_world[firstFoot[i][0]][firstIndex]
                dataNext1 = self.pose_world[nextFoot[i][0]][nextIndex]
                dataFirst2 = self.pose_world[firstFoot[i + 1][0]][firstIndex]

                data_pelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                data_pelvic2 = self.pose_world[firstFoot[i + 1][0]][pelvicIndex]

                [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[1], data_pelvic2[0], data_pelvic2[1])

                perp_slope = 1 / plineSlope

                [firstFootSlope1, firstFootIntercept1] = point_slope(perp_slope, dataFirst1[0], dataFirst1[1])
                [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext1[0], dataNext1[1])
                [firstFootSlope2, firstFootIntercept2] = point_slope(perp_slope, dataFirst2[0], dataFirst2[1])

                [firstInterceptX1, firstInterceptY1] = intercept(plineSlope, plineIntercept, firstFootSlope1, firstFootIntercept1)
                [nextInterceptX, nextInterceptY] = intercept(plineSlope, plineIntercept, nextFootSlope, nextFootIntercept)
                [firstInterceptX2, firstInterceptY2] = intercept(plineSlope, plineIntercept, firstFootSlope2, firstFootIntercept2)

                lengthsFirst[i] = dist(nextInterceptX, nextInterceptY, firstInterceptX2, firstInterceptY2)
                lengthsNext[i] = dist(firstInterceptX1, firstInterceptY1, nextInterceptX, nextInterceptY)

            else:
                dataFirst1 = self.pose_world[firstFoot[i][0]][firstIndex]
                dataNext1 = self.pose_world[nextFoot[i][0]][nextIndex]

                data_pelvic1 = self.pose_world[firstFoot[i][0]][pelvicIndex]
                data_pelvic2 = self.pose_world[firstFoot[i + 1][0]][pelvicIndex]

                [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[1], data_pelvic2[0], data_pelvic2[1])

                perp_slope = 1 / plineSlope

                [firstFootSlope1, firstFootIntercept1] = point_slope(perp_slope, dataFirst1[0], dataFirst1[1])
                [nextFootSlope, nextFootIntercept] = point_slope(perp_slope, dataNext1[0], dataNext1[1])

                [firstInterceptX1, firstInterceptY1] = intercept(plineSlope, plineIntercept, firstFootSlope1, firstFootIntercept1)
                [nextInterceptX, nextInterceptY] = intercept(plineSlope, plineIntercept, nextFootSlope, nextFootIntercept)

                lengthsNext[i] = dist(firstInterceptX1, firstInterceptY1, nextInterceptX, nextInterceptY)



                
     
    
        

        #If there is a mis-match in footfall count, we will calculate an extra step Length for the start foot. 
        #Note that it cannot be the case that there are more footsteps of the next foot, since this would imply that 
        #the next foot has 2 or more identified footfalls than the first foot
    
        if (misMatch):
            dataLast = self.pose_world[firstFoot[len(minFoot)][0]][firstIndex]
            dataPrior = self.pose_world[nextFoot[len(minFoot) - 1][0]][nextIndex]

            data_pelvic1 = self.pose_world[firstFoot[len(minFoot)][0]][firstIndex]
            data_pelvic2 = self.pose_world[nextFoot[len(minFoot) - 1][0]][nextIndex]

            [plineSlope, plineIntercept] = slope_intercept(data_pelvic1[0], data_pelvic1[1], data_pelvic2[0], data_pelvic2[1])

            perp_slope = 1 / plineSlope

            [lastFootSlope, lastFootIntercept] = point_slope(perp_slope, dataLast[0], dataLast[1])
            [priorFootSlope, priorFootIntercept] = point_slope(perp_slope, dataPrior[0], dataPrior[1])

            [lastInterceptX, lastInterceptY] = intercept(plineSlope, plineIntercept, lastFootSlope, lastFootIntercept)
            [priorInterceptX, priorInterceptY] = intercept(plineSlope, plineIntercept, priorFootSlope, priorFootIntercept)

            lengthsFirst[len(lengthsFirst)] = dist(priorInterceptX, priorInterceptY, lastInterceptX, lastInterceptY)

        totalValFirst = 0
        for val in lengthsFirst:
            totalValLeft += val

        totalValNext = 0
        for val in lengthsNext:
            totalValRight += val

        if (firstFoot == frameIDsLeft):
            return [totalValFirst/len(lengthsFirst), totalValNext/len(lengthsNext)]
        else:
            return [totalValNext/len(lengthsNext), totalValFirst/len(lengthsFirst)]
        



test = paramExtract('/Users/lukakoll/Downloads/wham_output.pkl')


        




        
