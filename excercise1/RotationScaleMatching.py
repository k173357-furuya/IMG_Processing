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
        F = function(beta)
        if F < 3:
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

        #特徴抽出機の生成
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
    print('imgFeaturePos',imgFeaturePos)

    distance = 0
    for i in range(2):
        minDisPos = returnMinVectorPos(imgFeaturePos[1][i],imgFeaturePos[0])
        #minDisPos = returnMinVectorPos(imgFeaturePos[0][i],imgFeaturePos[1])
        print('mindisPos',minDisPos)
        distance += np.linalg.norm(imgFeaturePos[1][i]-minDisPos)
    print('mindist',distance)


    #distance = np.linalg.norm(imgFeaturePos[0][0]-imgFeaturePos[1][0])+np.linalg.norm(imgFeaturePos[0][1]-imgFeaturePos[1][1])

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
    beta = np.array([0,1.2])
    #objectiveFunction(beta)
    rst=gaussNewton(objectiveFunction, beta,  1e-4, 1e-1)
    print('result',rst)

    img1 = cv2.imread('input-2019.png')
    img2 = cv2.imread('output-2019.png')

    imgHight, imgWeight,imgSize = img1.shape

    mat = cv2.getRotationMatrix2D(center=(475, 475), angle=rst[0], scale=rst[1])
    affine_img = cv2.warpAffine(img1, mat, (imgWeight, imgHight))
    cv2.imwrite('affine.jpg', affine_img)


    for i in range(2):
        if i == 0:
            img = img1
        if i == 1:
            img = affine_img


        #特徴抽出機の生成
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,3,3,0.235)

        #result is dilated for marking the corners, not important
        #dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[0,0,255]
        #print(img1[dst>0.01*dst.max()])
        cv2.imshow('dst',img)

        # Detect red pixel
        coord = np.where(np.all(img == (0, 0, 255), axis=-1))

        # Print coordinate
        print('img',i)
        print()
        for i in range(len(coord[0])):
            print("X:%s Y:%s"%(coord[1][i],coord[0][i]))

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()




'''
filename = 'input-2019.png'
#filename = 'output-2019.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,3,3,0.235)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
coord = np.where(np.all(img == (0, 0, 255), axis=-1))
print('subpix')
for i in range(len(coord[0])):
    print("X:%s Y:%s"%(coord[1][i],coord[0][i]))
cv2.imwrite('subpixel5.png',img)

#detector = cv2.xfeatures2d.SIFT_create()
detector = cv2.xfeatures2d.SURF_create()
#kpは特徴的な点の位置 destは特徴を現すベクトル
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)
#特徴点の比較機
bf = cv2.BFMatcher()
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(des1,des2, k=2)
#割合試験を適用
good = []
match_param = 0.35
for m,n in matches:
    if m.distance < match_param*n.distance:
        good.append([m])
#cv2.drawMatchesKnnは適合している点を結ぶ画像を生成する
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None,flags=2)

#cv2.imwrite("shift_result_Lab.png", img3)
cv2.imwrite("surf_result_Lab.png", img3)
'''
