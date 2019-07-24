#-------------------------------------------------------------------------------
# @Name : LSM_MLE.py
# @Abstract : Estimate Ellipse parameter from pixel position data by LSM and MLE
# @Author : Koki Furuya
# @Last edid : 2019/07/24
# reference : 菅谷先生の論文 http://www.iim.cs.tut.ac.jp/~kanatani/papers/hyperellip.pdf
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
def array_info(x):
    print("Array shape:", x.shape)
    print("Type of data:", x.dtype)
    print("Element of Array:\n",x,"\n")

#add gausserror
#正確な楕円上の位置(x,y)データ群に対して,標準偏差stdvの誤差を付与
def addGaussError(data, avg, stdv, absmax):
  noise = np.random.normal(avg, stdv, data.shape)
  noise = np.clip(noise, -(absmax + avg), absmax + avg)
  dataWError = data + noise
  return dataWError

#create MMatrix
#最小二乗法と最尤推定法(FNS)の双方で使うM行列の計算(reference参照)
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

#最尤推定用のL行列計算(FNS法で用いる)
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

#正規化共分散行列の計算
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

#LSM(最小二乗法による計算)
def estimateLSM(data):
    weight = np.matrix(np.full(data.shape[0],1.0)).T
    MMat = createMMat(data,weight)
    #CovMat = createCovMat(data_with_f0)
    #calculate EigenValue and EigenVector of minimum EigenValue
    #固有値を計算し, 最小固有値に対応する固有ベクトルを求める
    #この最小固有値がMを最小にするベクトル → 最小二乗法の解
    #la,v = np.linalg.eig(MMat)
    la,v = sp.linalg.eigh(MMat)
    #print(la)
    #la,v = np.linalg.eigvals(MMat)
    myu = np.matrix(v[:, np.argmin(np.absolute(la))]).T
    #index = np.where(la==min(la))[0][0]
    #myu=np.matrix(v[:, index]).T
    if myu.sum()<0:
        myu=-myu
    return myu

#FNS(最尤推定法による計算)
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
    covs = []
    for i in range(dataMod.shape[0]):
        data_row = dataMod[i, :]
        covs.append(createCovMat(data_row))

    loop = 0
    while True:
        # M Matrix
        M = createMMat(dataMod, weight)
        L = createLMat(dataMod, weight, myu, covs)
        lamdas, v = sp.linalg.eigh((M - L),turbo=False)
        myuOrg = myu
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
        #array_info(weight)
    if myu.sum()<0:
        myu = -myu

    return myu

#分散の計算
def calcDeviation(results,true_val):
    sum_theta = np.matrix(np.zeros((6,1)))
    for rst in results:
        p_theta = np.matrix(np.identity(6)) - np.dot(true_val,true_val.T)
        delta_theta = np.dot(p_theta,rst)
        #array_info(delta_theta)
        sum_theta = sum_theta + delta_theta
    sum_theta = sum_theta/len(results)
    rms_value = np.linalg.norm(sum_theta)
    return rms_value

#RMS誤差の計算
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

#KCR下(理論上, それ以下の精度は出せないという値)の計算
def KCR_lower_bound(data,stdv,data_num,myu):

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

#show 2D Ellipse image
def plotData(myuLSM,myuFNS,trueVal,data):
    import sys
    from Ellipse import generateVecFromEllipse
    from Ellipse import getEllipseProperty

    trueVal = np.matrix(trueVal).T
    myu = np.matrix(np.zeros(6)).T
    fig, ax = plt.subplots(ncols = 1, figsize=(10, 10))
    label_text = ''

    for i in range(3):
        if(i==0):
            myu = myuLSM
            label_text = 'LSM'
        if(i==1):
            myu = myuFNS
            label_text = 'FNS'
        if(i==2):
            myu = trueVal
            label_text = 'TrueAns'

        valid, axis, centerEst, Rest = getEllipseProperty(myu[0,0], myu[1,0], myu[2,0], myu[3,0], myu[4,0], myu[5,0])
        dataEst = generateVecFromEllipse(axis, centerEst, Rest)
        ax.plot(dataEst[:, 0], dataEst[:, 1])
    ax.scatter(data[:,0]/600,data[:,1]/600)
    ax.legend(['LSM','FNS','TrueAns'])

    plt.savefig('Ellipse.png')
    return 0

def calcLSMandMSE(data,trialNum ,stdv_val,trueAns):
    f0=600
    stdv = stdv_val #standard Error
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

    LSM_dev = calcDeviation(LSM_results,np.matrix(trueAns).T)
    FNS_dev = calcDeviation(FNS_results,np.matrix(trueAns).T)
    LSM_err = calcRMSErr(LSM_results,np.matrix(trueAns).T)
    FNS_err = calcRMSErr(FNS_results,np.matrix(trueAns).T)
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

    return LSM_dev ,FNS_dev,LSM_err,FNS_err,kcr

#LSMとMSEを繰り返し計算し, その誤差平均値などを求める.
#実質main関数
def calcRepLSMandMSE():
    #read data of points including error value
    data = np.loadtxt('points.dat',comments='!')
    #read data of true value
    trueAns = np.loadtxt('true_param.dat')
    #init value
    trialNum = 1000
    #List of standard deviation
    stdvList = np.arange(0.0, 0.121, 0.005, dtype = 'float64')
    #Make list of SD List and RMS list
    LSM_dev_List = np.array(np.zeros(stdvList.shape))
    MSE_dev_List = np.array(np.zeros(stdvList.shape))
    LSM_err_List = np.array(np.zeros(stdvList.shape))
    MSE_err_List = np.array(np.zeros(stdvList.shape))
    KCR_List = np.array(np.zeros(stdvList.shape))

    for i in range(stdvList.shape[0]):
        LSM_dev_List[i],MSE_dev_List[i],LSM_err_List[i],MSE_err_List[i],KCR_List[i] = calcLSMandMSE(data,trialNum ,stdvList[i],trueAns)

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

    calcRepLSMandMSE()

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
