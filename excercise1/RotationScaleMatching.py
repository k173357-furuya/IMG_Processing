import numpy as np
import cv2
#img1 = cv2.imread('Lab1_VGA.JPG',1)
#img2 = cv2.imread('Lab2_VGA.JPG',1)
img1 = cv2.imread('input-2019.png',0)
img2 = cv2.imread('output-2019.png',0)
#特徴抽出機の生成
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
