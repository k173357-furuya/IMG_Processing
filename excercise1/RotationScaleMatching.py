import numpy as np
import cv2
import matplotlib.pyplot as plt

#参考
#https://org-technology.com/posts/gauss-newton-method.html

def gaussNewton(function, beta, tolerance, epsilon):
    # Gauss-Newton 法
    delta = 2*tolerance
    alpha = 1
    while np.linalg.norm(delta) > tolerance:
        print('beta',beta)
        F = function(beta)
        if F < 2:
            break
        J = np.zeros((len(F), len(beta)))  # 有限差分ヤコビアン
        for jj in range(0, len(beta)):
            dBeta = np.zeros(beta.shape)
            dBeta[jj] = epsilon
            J[:, jj] = (function(beta+dBeta)-F)/epsilon
        delta = -np.linalg.pinv(J).dot(F)  # 探索方向
        beta = beta + alpha*delta
    return beta

def objectiveFunction(beta):
    inputPos = np.array([[705,301],[306,435]])
    outputPos = np.array([[683,180],[259,493]])
    centerPos = np.array([475,475])

    piDeg2Rad = (1.0/180) * np.pi
    radTheta = beta[0]*piDeg2Rad
    print('beta',beta)
    MMat = beta[1]*np.matrix([[np.cos(radTheta),np.sin(radTheta)],[-np.sin(radTheta),np.cos(radTheta)]])

    RotadScaledPosList = []
    for i in range(2):
        pos = MMat * np.matrix(inputPos[i]-centerPos).T
        pos = pos + np.matrix(centerPos).T
        RotadScaledPosList.append(pos)

    distance = 0
    for i in range(2):
        nowPos = np.array([RotadScaledPosList[i][0],RotadScaledPosList[i][1]])
        print('nowPos',nowPos)
        #distance += np.linalg.norm(outputPos[i]-nowPos)
        distance += (outputPos[i][0]-nowPos[0])**2+(outputPos[i][1]-nowPos[1])**2
    print('distance',distance[0][0])

    return np.array([distance[0][0]])



def objectiveFunction_test(beta):
    img1 = cv2.imread('input-2019.png')
    img2 = cv2.imread('output-2019.png')

    imgHight, imgWeight,imgSize = img1.shape
    mat = cv2.getRotationMatrix2D(center=(475, 475), angle=beta[0], scale=beta[1])
    affine_img = cv2.warpAffine(img1, mat, (imgWeight, imgHight))

    imgFeaturePos = []
    for i in range(2):
        if i == 0:
            img = affine_img
        if i == 1:
            img = img2

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        if i==0:
            dst = cv2.cornerHarris(gray,3,3,0.235)
        else:
            dst = cv2.cornerHarris(gray,3,3,0.235)
        img[dst>0.01*dst.max()]=[0,0,255]

        #cv2.imshow('dst',img)

        # Detect red pixel
        coord = np.where(np.all(img == (0, 0, 255), axis=-1))

        # Print coordinate
        print('img',i)
        print()
        posList = []
        for i in range(len(coord[0])):
            posList.append(np.array([coord[1][i],coord[0][i]]))
            print("X:%s Y:%s"%(coord[1][i],coord[0][i]))

        imgFeaturePos.append(posList)


    # 目的関数
    #print('imgFeaturePos',imgFeaturePos)

    distance = 0
    for i in range(2):
        minDisPos = returnMinVectorPos(imgFeaturePos[1][i],imgFeaturePos[0])
        print('mindisPos',minDisPos)
        #distance = distance + (minDisPos[0] - imgFeaturePos[1][i][0])**2
        #distance = distance + (minDisPos[1] - imgFeaturePos[1][i][1])**2
        distance += np.linalg.norm(imgFeaturePos[1][i]-minDisPos)

    print('distance',distance)
    #distance = imgFeaturePos[0]
    #r = y - theoreticalValue(beta)
    return np.array([distance])

def returnMinVectorPos(TruePos,VecList):
    minVal = 100000
    nearPos = []
    for i in range(len(VecList)):
        if minVal > np.linalg.norm(TruePos-VecList[i]):
            minVal = np.linalg.norm(TruePos-VecList[i])
            nearPos = VecList[i]
    return nearPos

def theoreticalValue(beta):
    img2 = cv2.imread('output-2019.png')
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,3,3,0.235)
    img[dst>0.01*dst.max()]=[0,0,255]
    coord = np.where(np.all(img == (0, 0, 255), axis=-1))

    posList = []
    for i in range(len(coord[0])):
        posList.append(np.array([coord[1][i],coord[0][i]]))
        #print("X:%s Y:%s"%(coord[1][i],coord[0][i]))

    # 理論値
    f = beta[0]*x / (beta[1]+x)
    return f

if __name__ == "__main__":
    beta = np.array([0,1.0])
    rst=gaussNewton(objectiveFunction, beta,  1e-4, 1e-4)
    print('result',rst)

    #objectiveFunction_test(beta)

    img1 = cv2.imread('input-2019.png')
    img2 = cv2.imread('output-2019.png')

    imgHight, imgWeight,imgSize = img1.shape

    mat = cv2.getRotationMatrix2D(center=(475, 475), angle=rst[0], scale=rst[1])
    print('affin_matrix',mat)
    affine_img = cv2.warpAffine(img1, mat, (imgWeight, imgHight))
    cv2.imwrite('affine.jpg', affine_img)


    for i in range(2):
        if i == 0:
            img = img1
        if i == 1:
            img = img2


        #特徴抽出機の生成
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,3,3,0.235)

        #result is dilated for marking the corners, not important
        #dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[0,0,255]

        #drawCircle for featrure point
        coord = np.where(np.all(img == (0, 0, 255), axis=-1))
        #for j in range(len(coord[0])):
            #cv2.circle(img, (coord[1][j], coord[0][j]), 15, (0, 0, 255), thickness=-1)
        #cv2.imwrite('img'+str(i+1)+'_FeaturePoint.jpg', img)

        #print(img1[dst>0.01*dst.max()])
        cv2.imshow('dst',img)

        # Detect red pixel
        #coord = np.where(np.all(img == (0, 0, 255), axis=-1))

        # Print coordinate
        print('img',i)
        print()
        for j in range(len(coord[0])):
            print("X:%s Y:%s"%(coord[1][j],coord[0][j]))

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
