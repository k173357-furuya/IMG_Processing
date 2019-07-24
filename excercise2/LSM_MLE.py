# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# @Name : LSM_MLE.py
# @Abstract : Estimate Ellipse parameter from pixel position data by LSM and MLE
# @Author : Koki Furuya
# @Last edid : 2019/07/24
# reference : 菅谷先生の論文 http://www.iim.cs.tut.ac.jp/~kanatani/papers/hyperellip.pdf
#           : 楕円あてはめのプログラム例 http://daily-tech.hatenablog.com/entry/2018/04/15/001043
#-------------------------------------------------------------------------------
import numpy as np
from scipy import linalg
import scipy as sp
import matplotlib.pyplot as plt
import math
from sympy import *
import sympy
import scipy.sparse.linalg

#show array infomation of vector or matrix
#================================================
#weight:重み
#myu:楕円の推定パラメータ
#covs:共分散行列
#
#正確な楕円上の位置(x,y)データ群に対して,標準偏差stdvの誤差を付与
#================================================
def array_info(x):
    print("Array shape:", x.shape)
    print("Type of data:", x.dtype)
    print("Element of Array:\n",x,"\n")

#================================================
#addGaussError(data, avg, stdv, absmax):
#data:楕円の点群座標
#avg:正規分布の平均値
#stdv:標準偏差
#absmax:これ以上の値がでたら丸める.
#
#正確な楕円上の位置(x,y)データ群に対して,標準偏差stdvの誤差を付与
#================================================
def addGaussError(data, avg, stdv, absmax):
  noise = np.random.normal(avg, stdv, data.shape)
  noise = np.clip(noise, -(absmax + avg), absmax + avg)
  dataWError = data + noise
  return dataWError

#================================================
#createLMat(data, weight, myu, covs):
#data:一つの点のx,y,f0パラメータ
#weight:重み
#myu:楕円の推定パラメータ
#covs:共分散行列
#
#最小二乗法と最尤推定法(FNS)の双方で使うM行列の計算(reference参照)
#================================================
def createMMat(data, weight):

  dataMod = np.matrix(np.zeros((data.shape[0], 6)))
  for i in range(data.shape[0]):
    dataMod[i, 0] = data[i, 0]**2
    dataMod[i, 1] = data[i, 0] * data[i, 1]
    dataMod[i, 2] = data[i, 1]**2
    dataMod[i, 3] = data[i, 0] * data[i, 2]
    dataMod[i, 4] = data[i, 1] * data[i, 2]
    dataMod[i, 5] = data[i, 2]**2

  M = np.zeros((6, 6))
  M = np.matrix(M)

  for i in range(dataMod.shape[0]):
    dM = dataMod[i, :].T * dataMod[i, :]
    M = M + weight[i, 0] * dM

  return M / dataMod.shape[0]

#================================================
#createLMat(data, weight, myu, covs):
#data:一つの点のx,y,f0パラメータ
#weight:重み
#myu:楕円の推定パラメータ
#covs:共分散行列
#
#最尤推定用のL行列計算(FNS法で用いる).論文でもLで表記
#================================================
def createLMat(data, weight, myu, covs):

  dataMod = np.matrix(np.zeros((data.shape[0], 6)))
  for i in range(data.shape[0]):
    dataMod[i, 0] = data[i, 0]**2
    dataMod[i, 1] = data[i, 0] * data[i, 1]
    dataMod[i, 2] = data[i, 1]**2
    dataMod[i, 3] = data[i, 0] * data[i, 2]
    dataMod[i, 4] = data[i, 1] * data[i, 2]
    dataMod[i, 5] = data[i, 2]**2

  L = np.matrix(np.zeros((6, 6)))
  for i in range(dataMod.shape[0]):
    coeff = weight[i, 0]**2 * (dataMod[i, :] * myu)**2
    L = L + coeff[0, 0] * covs[i]

  return L / dataMod.shape[0]

#================================================
#estimateFNS(data):
#data:一つの点のx,y,f0パラメータ
#
#正規化共分散行列の計算.Vol[ξ]のやつ
#================================================
def createCovMat(data):

    x = data[0, 0]
    y = data[0, 1]
    f0 = data[0, 2]
    xx = x**2
    yy = y**2
    xy = x*y
    f0x = f0*x
    f0y = f0*y
    f0f0 = f0**2


    cov = np.matrix([[xx,  xy,     0,   f0x,     0,    0], \
                   [xy,  xx+yy, xy,   f0y,   f0x,    0], \
                   [0,   xy,    yy,     0,   f0y,    0], \
                   [f0x, f0y,   0,    f0f0,    0,    0], \
                   [0,   f0x,   f0y,  0,     f0f0,   0], \
                   [0,   0,     0,    0,     0,      0]])

    #cov = cov
    return 4*cov


#================================================
#estimateFNS(data):
#data:楕円の点群座標
#
#最小二乗法(LSM)による計算.
#================================================
def estimateLSM(data):
    weight = np.matrix(np.full(data.shape[0],1.0)).T
    MMat = createMMat(data,weight)
    #calculate EigenValue and EigenVector of minimum EigenValue
    #固有値を計算し, 最小固有値に対応する固有ベクトルを求める
    #この最小固有値がMを最小にするベクトル → 最小二乗法の解
    #la,v = np.linalg.eig(MMat)
    la,v = sp.linalg.eigh(MMat)
    myu = np.matrix(v[:, np.argmin(np.absolute(la))]).T
    if myu.sum()<0:
        myu=-myu
    return myu


#================================================
#estimateFNS(data):
#data:楕円の点群座標
#
#最尤推定法(FNS)による計算.
#================================================
def estimateFNS(data):
    dataMod = data
    # Param Vector
    myu = np.matrix(np.zeros(6)).T
    myuNew = myu
    myuOrg = myu
    # Weight matrix.
    weight = np.ones(dataMod.shape[0])
    weight = np.matrix(weight).T
    # Covars
    # 共分散行列: 各座標のベクトルごとに共分散行列を作成
    covs = []
    for i in range(dataMod.shape[0]):
        data_row = dataMod[i, :]
        covs.append(createCovMat(data_row))

    #FNS法による最適パラメータの探索
    loop = 0
    while True:
        # M Matrix
        M = createMMat(dataMod, weight)
        L = createLMat(dataMod, weight, myu, covs)
        #固有値,固有ベクトルの計算
        lamdas, v = sp.linalg.eigh((M - L),turbo=False)
        myuOrg = myu
        #最小固有値に対する固有ベクトルを計算.それが新しい楕円パラメータ推定値になる.
        index = [i for i, v in enumerate(lamdas) if v == min(lamdas)][0]
        myuNew=np.matrix(v[:, index]).T
        if myuNew.sum()<0:
            myuNew=-myuNew

        myu = myuNew

        term = np.linalg.norm(np.absolute(myu) - np.absolute(myuOrg))
        if term < 10e-6 or loop > 100:
            if loop > 100:
                print('loop > 100')
            break

        #weightの更新
        for i in range(dataMod.shape[0]):
            alp = myu.T * covs[i] * myu
            weight[i, 0] = 1 / (alp)

        loop = loop + 1
    if myu.sum()<0:
        myu = -myu

    return myu


#================================================
#calcRMSErr(results,true_val):
#results:楕円推定のパラメータ(全繰り返し分)
#true_val:パラメータの正解値
#
#分散の計算
#================================================
def calcDeviation(results,true_val):
    sum_theta = np.matrix(np.zeros((6,1)))
    for rst in results:
        p_theta = np.matrix(np.identity(6)) - np.dot(true_val,true_val.T)
        delta_theta = np.dot(p_theta,rst)
        sum_theta = sum_theta + delta_theta
    sum_theta = sum_theta/len(results)
    rms_value = np.linalg.norm(sum_theta)
    return rms_value


#================================================
#calcRMSErr(results,true_val):
#results:楕円推定のパラメータ(全繰り返し分)
#true_val:パラメータの正解値
#
#RMS誤差の計算. 計算方法が少し特殊なので論文参照
#================================================
def calcRMSErr(results,true_val):
    sum_theta = 0.0
    for rst in results:
        p_theta = np.identity(6) - np.dot(true_val,true_val.T)
        delta_theta = np.dot(p_theta,rst)
        sum_theta = sum_theta + np.linalg.norm(delta_theta)**2
    rms_value = np.sqrt(sum_theta/len(results))

    testsum = 0.0
    for rst in results:
        testsum = testsum + np.linalg.norm(rst - true_val)
    print('norm distance:',testsum/len(results))
    return rms_value

#================================================
#KCR_lower_bound(data,stdv,myu):
#data:楕円の点群座標
#stdv:標準偏差
#myu:楕円パラメータの正解の値
#
#KCR下界(理論上, それ以下の精度は出せないという値)の計算
#================================================
def KCR_lower_bound(data,stdv,myu):
    # Weight matrix.
    weight = np.ones(data.shape[0])
    weight = np.matrix(weight).T

    covs = []
    for i in range(data.shape[0]):
        data_row = data[i, :]
        covs.append(createCovMat(data_row))

    for i in range(data.shape[0]):
        alp = myu.T * covs[i] * myu
        weight[i, 0] = 1 / (alp)

    M = createMMat(data, weight)
    lamda,v = np.linalg.eigh(M)

    sorted_lamda = sorted(lamda, reverse=True)
    print('sorted_lamda',sorted_lamda)
    sorted_lamda = np.matrix(sorted_lamda[:5]).T

    sum_inv_ramda = 0.0
    for i in range(sorted_lamda.shape[0]):
        sum_inv_ramda = sum_inv_ramda + (1.0/sorted_lamda[i])

    Dkcr = stdv*np.sqrt(sum_inv_ramda) / np.sqrt(data.shape[0])
    return Dkcr[0][0]


#================================================
#plotData(myuLSM,myuFNS,trueVal,data):
#myuLSM:LSMの推定楕円パラメータ
#myuFNS:FNSの推定楕円パラメータ
#trueVal:パラメータの真値
#data:楕円の点群座標
#
#楕円の形状を描画
#================================================
def plotData(myuLSM,myuFNS,trueVal,data):
    import sys
    from Ellipse import generateVecFromEllipse
    from Ellipse import getEllipseProperty

    trueVal = np.matrix(trueVal).T
    myu = np.matrix(np.zeros(6)).T
    fig, ax = plt.subplots(ncols = 1, figsize=(10, 10))

    for i in range(3):
        if(i==0):
            myu = myuLSM
        if(i==1):
            myu = myuFNS
        if(i==2):
            myu = trueVal

        valid, axis, centerEst, Rest = getEllipseProperty(myu[0,0], myu[1,0], myu[2,0], myu[3,0], myu[4,0], myu[5,0])
        dataEst = generateVecFromEllipse(axis, centerEst, Rest)
        ax.plot(dataEst[:, 0], dataEst[:, 1])
    ax.scatter(data[:,0]/600,data[:,1]/600)
    ax.legend(['LSM','FNS','TrueAns'])

    plt.savefig('Ellipse.png')
    return 0

#================================================
#calcLSMandMSE(data,trialNum ,stdv_val,trueAns):
#data:点群座標
#trialNum:繰り返し回数
#stdv_val:標準偏差
#trueAns:正解の楕円パラメータ
#
#戻り値: LSMの誤差,偏差, MSEの誤差, 偏差
#LSMとMSEで楕円のパラメータを推定し, その誤差,偏差を返す
#================================================
def calcLSMandMSE(data,trialNum ,stdv_val,trueAns):
    f0=600 #倍率
    stdv = stdv_val #標準偏差
    LSM_results = [] #最小二乗法の結果
    FNS_results = [] #最尤推定(FNS)の結果
    myu = np.matrix([1.0,1.0,1.0,1.0,1.0,1.0]).T #楕円パラメータ
    f_exp2 = np.matrix(np.full(data.shape[0],f0)) #f0のベクトル化

    #試行回数まで繰り返し最小二乗法と最尤推定を実施
    for i in range(trialNum):
        #show index
        print('loop :',i+1)
        #add Gaussian noise
        #標準偏差stdvの誤差を点群の座標に付与
        dataNoised = addGaussError(data, 0, stdv, 100)
        #create M matrix
        data_with_f0 = np.concatenate((np.matrix(dataNoised),f_exp2.T),axis = 1)
        #calculate LSM
        myuLSM = estimateLSM(data_with_f0)
        #calculate FNS
        myuFNS = estimateFNS(data_with_f0)

        #make results list
        LSM_results.append(myuLSM)
        FNS_results.append(myuFNS)

    LSM_dev = calcDeviation(LSM_results,np.matrix(trueAns).T)
    FNS_dev = calcDeviation(FNS_results,np.matrix(trueAns).T)
    LSM_err = calcRMSErr(LSM_results,np.matrix(trueAns).T)
    FNS_err = calcRMSErr(FNS_results,np.matrix(trueAns).T)
    kcr = KCR_lower_bound(np.concatenate((np.matrix(data),f_exp2.T),axis = 1),stdv ,np.matrix(trueAns).T)
    print('LSM_uの値',myuLSM)
    print('FNS_uの値',myuFNS)
    print('真値',np.matrix(trueAns).T)
    print('真値とLSM_uの誤差:',LSM_err)
    #print(calcVectorError(myu,np.matrix(trueAns).T))
    print(np.linalg.norm(myuLSM - np.matrix(trueAns).T))
    print('真値とFNS_uの誤差:',FNS_err)
    print(np.linalg.norm(myuFNS - np.matrix(trueAns).T))
    print('真値とLSM_uの偏差:',LSM_dev)
    print('真値とFNS_uの偏差:',FNS_dev)
    print('KCR誤差',kcr)

    return LSM_dev ,FNS_dev,LSM_err,FNS_err,kcr

#================================================
#実質main関数
#LSMとMSEを繰り返し計算し, その誤差平均値などを求める
#================================================
def calcRepLSMandMSE():
    #read data of points including error value
    #データ点(x,y)座標の集まりを読み込み
    data = np.loadtxt('points.dat',comments='!')
    #read data of true value
    #推定パラメータ(A,B,C,D,E,F)の真値を読み込み
    trueAns = np.loadtxt('true_param.dat')

    #---------------------------------------------------
    #値の初期化
    #----------------------------------------------------
    # 一つのstdv(標準偏差)に対して, trialNum回だけ繰り返し実行する
    trialNum = 1000
    #List of standard deviation
    #標準偏差のベクトル stdvList = np.arrange(初期値, 終了値, 変化量, 値の型)
    #初期値から終了値までstdv(標準偏差)を変化させながら, trialNum回だけ繰り返して結果の平均を出す
    stdvList = np.arange(0.0, 0.121, 0.005, dtype = 'float64')
    #Make list of SD List and RMS list
    #各結果を格納するためのリスト
    LSM_dev_List = np.array(np.zeros(stdvList.shape)) #LSMの結果の偏差
    MSE_dev_List = np.array(np.zeros(stdvList.shape)) #MSE(FNS)の結果の偏差
    LSM_err_List = np.array(np.zeros(stdvList.shape)) #LSMの結果のRMS誤差
    MSE_err_List = np.array(np.zeros(stdvList.shape)) #MSE(FNS)の結果のRMS誤差
    KCR_List = np.array(np.zeros(stdvList.shape)) #KCR下界(理論上これ以下にはならない値)

    #---------------------------------------------------
    #繰り返しFNSとMSEを計算
    #----------------------------------------------------
    for i in range(stdvList.shape[0]):
        LSM_dev_List[i],MSE_dev_List[i],LSM_err_List[i],MSE_err_List[i],KCR_List[i] = calcLSMandMSE(data,trialNum ,stdvList[i],trueAns)

    #---------------------------------------------------
    #結果の描画と画像書き出し
    #----------------------------------------------------
    #plot all results
    plt.rcParams['font.family'] ='sans-serif'#使用するフォント
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 8 #フォントの大きさ
    plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
    #plot deviation
    plt.plot(stdvList,LSM_dev_List,label='LSM')
    plt.plot(stdvList,MSE_dev_List,label='MSE')
    plt.xlim([np.min(stdvList),np.max(stdvList)])
    plt.ylim([0,plt.ylim()[1]])
    plt.legend()
    plt.savefig('deviation.png')
    #plot RMS
    plt.figure()
    plt.plot(stdvList,LSM_err_List,label='LSM')
    plt.plot(stdvList,MSE_err_List,label='MSE')
    plt.plot(stdvList,KCR_List,linestyle="--",label='KCR Lower Bound')
    plt.xlim([np.min(stdvList),np.max(stdvList)])
    plt.ylim([0,plt.ylim()[1]])
    #plt.xlabel("Standard deviation")
    #plt.ylabel("Y-axis")
    plt.legend()
    plt.savefig('RMS.png')

if __name__ == "__main__":

#この関数で全て実施,パラメータの設定もこの関数の中で実施
    calcRepLSMandMSE()

#--------------------------------------------------------------------------------
# デバッグ用 : 指定した一つのstdv(標準偏差)の結果だけ見たい場合は以下のプログラムを使用すればいい
# -------------------------------------------------------------------------------
    '''
    #read data of points including error value
    data = np.loadtxt('points.dat',comments='!')
    #read data of true value
    trueAns = np.loadtxt('true_param.dat')
    #init value
    trialNum  = 10

    f0=600
    stdv = 0.05 #standard Error
    LSM_results = []
    FNS_results = []
    myu = np.matrix([1.0,1.0,1.0,1.0,1.0,1.0]).T
    f_exp2 = np.matrix(np.full(data.shape[0],f0))


    for i in range(trialNum ):
        #show index
        print('loop :',i+1)
        #add Gaussian noise
        dataNoised = addGaussError(data, 0, stdv, 100)
        #create M matrix
        data_with_f0 = np.concatenate((np.matrix(dataNoised),f_exp2.T),axis = 1)
        #calculate LSM
        myuLSM = estimateLSM(data_with_f0)
        #calculate FNS
        myuFNS = estimateFNS(data_with_f0)

        #make results list
        LSM_results.append(myuLSM)
        FNS_results.append(myuFNS)
    #calculate Taubin
    #myuTaubin = estimateTaubin(data_with_f0,weight)
    #print(myuTaubin)
    #calculate FNS myu
    #calc Error of results
    #print('FMS_resutls',FNS_results)
    plotData(myuLSM,myuFNS,trueAns,dataNoised)

    #plotData(dataEst,dataEst)


    LSM_dev = calcDeviation(LSM_results,np.matrix(trueAns).T)
    FNS_dev = calcDeviation(FNS_results,np.matrix(trueAns).T)
    LSM_err = calcRMSErr(LSM_results,np.matrix(trueAns).T)
    FNS_err = calcRMSErr(FNS_results,np.matrix(trueAns).T)
    #kcr = KCR_lower_bound(np.concatenate((np.matrix(trueAns).T,f_exp2.T),axis = 1),stdv,trialNum )
    kcr = KCR_lower_bound(np.concatenate((np.matrix(data),f_exp2.T),axis = 1),stdv,trialNum ,np.matrix(trueAns).T)
    print('LSM_uの値',myuLSM)
    print('FNS_uの値',myuFNS)
    print('真値',np.matrix(trueAns).T)
    print('真値とLSM_uの誤差:',LSM_err)
    #print(calcVectorError(myu,np.matrix(trueAns).T))
    print(np.linalg.norm(myuLSM - np.matrix(trueAns).T))
    print('真値とFNS_uの誤差:',FNS_err)
    print(np.linalg.norm(myuFNS - np.matrix(trueAns).T))
    print('真値とLSM_uの偏差:',LSM_dev)
    print('真値とFNS_uの偏差:',FNS_dev)
    print('KCR誤差',kcr)
    '''
