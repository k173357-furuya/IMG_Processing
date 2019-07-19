import numpy as np
from scipy import linalg
import scipy as sp
import matplotlib.pyplot as plt
import math
from sympy import *
import sympy
import scipy.sparse.linalg

def array_info(x):
    print("Array shape:", x.shape)
    print("Type of data:", x.dtype)
    print("Element of Array:\n",x,"\n")

#add gausserror
def addGaussError(data, avg, stdv, absmax):
  noise = np.random.normal(avg, stdv, data.shape)
  #print(noise)
  noise = np.clip(noise, -(absmax + avg), absmax + avg)
  dataWError = data + noise
  return dataWError

#create Matrix
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
  #array_info(weight)
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
    #array_info(coeff)
    #L = L + coeff[0, 0] * createCovMat(data[i,:])
    L = L + coeff[0, 0] * covs[i]
    #array_info(coeff[0, 0] * covs[i])
    #print('weight:',weight[i, 0])
    #print('weight**2:',weight[i, 0]**2)
    #print('(dataMod[i, :] * myu)**2:',(dataMod[i, :] * myu)**2)
    #print('coeff[0, 0] * covs[i]:',coeff[0, 0] * covs[i])

  return L / dataMod.shape[0]

#
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

#normalizetion of vector
def array_normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

def calcVectorError(Vct,trueVct):
    sumerr = 0.0
    for i in range(Vct.shape[0]):
        sumerr = sumerr + np.sqrt((Vct[i,0]-trueVct[i,0])**2)
    return sumerr

def calc_Weight_LSM(weight,data,myu,row):
    for i in range(data.shape[0]):
        CovMat = createCovMat(data[i,:])
        alp = myu.T * CovMat * myu
        weight[i,0] = 1/(alp)
        #Cov_myu = np.dot(CovMat,myu)
        #weight[i,0] = 1/(np.dot(Cov_myu.T,myu))
        #weight = np.matrix(np.full(row,weight_value)).T
    #array_info(weight)
    return weight

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

def estimateFNS(data):
    # Add normalized term.
    dataMod = data
    #dataMod = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

    # Param Vector
    myu = np.matrix(np.zeros(6)).T
    myuNew = myu
    myuOrg = myu

    # Weight matrix.
    weight = np.ones(dataMod.shape[0])
    weight = np.matrix(weight).T
    #myu = estimateLSM(data)
    # Covars
    covs = []
    for i in range(dataMod.shape[0]):
        data_row = dataMod[i, :]
        covs.append(createCovMat(data_row))
        #print(covs[i])
    #myu = estimateLSM(dataMod)
    loop = 0
    while True:
        # M Matrix
        M = createMMat(dataMod, weight)
        L = createLMat(dataMod, weight, myu, covs)
        #array_info(M)
        #array_info(L)
        #print('dataMod',dataMod)
        #print('weight',weight)
        #lamdas, v = np.linalg.eigh((M - L))
        lamdas, v = sp.linalg.eigh((M - L),turbo=False)
        #print('lamdas:',lamdas)
        #print('v:',v)

        myuOrg = myu

        #myuNew = np.matrix(v[:, np.argmin(np.absolute(lamdas))]).T
        #print(la)
        #la,v = np.linalg.eigvals(MMat)
        #index = np.where(lamdas==min(lamdas))[0][0]
        #sorted_lamdas = sorted(lamdas,reverse=True)[:5]
        #print(sorted_lamdas)
        #print(sorted_lamdas)
        #myuNew = np.matrix(v[:, np.argmin(sorted_lamdas)]).T
        #index = [i for i, v in enumerate(sorted_lamdas) if v == min(sorted_lamdas)][0]
        #print(v[:, index])
        index = [i for i, v in enumerate(lamdas) if v == min(lamdas)][0]
        myuNew=np.matrix(v[:, index]).T
        #print('lamda:',lamdas)
        #print('index:',np.argmin(np.absolute(lamdas)))
        #print('lamda:',lamdas)
        #print(v[:, index])

        #print( (M-L)*myuNew - np.matrix(np.diag(lamdas))*myuNew )
        #if loop==0:
            #myuNew = np.matrix(v[:, np.argmin(np.absolute(lamdas))]).T
        #myuNew = np.matrix(v[:, np.argmin(np.absolute(lamdas))])
        #myuNew = np.matrix(v[:, np.argmin(np.absolute(lamdas))]).T

        #print('resutl:',((M-L)-np.argmin(np.absolute(lamdas)*np.matrix(np.ones((6,6)))))*myuNew)
        #print('myuNew',myuNew)
        #myuNew = np.matrix(v[:, np.argmin(lamdas)]).T
        #print('index',index)
        #print('v',v)
        #print('lamdas',lamdas)
        #print('min(v)',np.min(lamdas))
        if myuNew.sum()<0:
            myuNew=-myuNew

        #myu = (myuNew + myuOrg) / 2
        myu = myuNew

        #term = np.linalg.norm(myu - myuOrg)

        term = np.linalg.norm(np.absolute(myu) - np.absolute(myuOrg))
        #print('term:',term)
        #print('term:',term)
        if term < 10e-6 or loop > 100:
            if loop > 100:
                print('loop > 100')
            break


        #weight = calc_Weight_LSM(weight,data,myu,data.shape[0])
        for i in range(dataMod.shape[0]):
            alp = myu.T * covs[i] * myu
            weight[i, 0] = 1 / (alp)

        loop = loop + 1
        #array_info(weight)
        calcFunctionJ(dataMod,covs,myu)
    if myu.sum()<0:
        myu = -myu

    #if myu[0,0]<0:
    return myu

def calcFunctionJ(data,covs,myu):
    dataMod = np.matrix(np.zeros((data.shape[0], 6)))
    for i in range(data.shape[0]):
       dataMod[i, 0] = data[i, 0]**2
       dataMod[i, 1] = data[i, 0] * data[i, 1]
       dataMod[i, 2] = data[i, 1]**2
       dataMod[i, 3] = data[i, 0] * data[i, 2]
       dataMod[i, 4] = data[i, 1] * data[i, 2]
       dataMod[i, 5] = data[i, 2]**2

    #calc J
    J=0.0
    for i in range(data.shape[0]):
        alp = myu.T * covs[i] * myu
        J = J + (myu.T*dataMod[i, :].T)**2 /alp
    #print('J:',J)
    #print('myu',myu)
    return J

def calcDeviation(results,true_val):
    sum_theta = np.matrix(np.zeros((6,1)))
    for rst in results:
        p_theta = np.matrix(np.identity(6)) - np.dot(true_val,true_val.T)
        delta_theta = np.dot(p_theta,rst)
        #array_info(delta_theta)
        sum_theta = sum_theta + delta_theta
    sum_theta = sum_theta/len(results)
    rms_value = np.linalg.norm(sum_theta)
    #rms_value = np.linalg.norm(sum_theta,2)
        #print(sum_theta)
        #print(np.linalg.norm(delta_theta,2)**2)
    #print('sum_theta/resultsNum:',sum_theta/len(results))
    #rms_value = np.sqrt(sum_theta/len(results))
    #print(rms_value)
    return rms_value

def calcRMSErr(results,true_val):
    sum_theta = 0.0
    for rst in results:
        p_theta = np.identity(6) - np.dot(true_val,true_val.T)
        delta_theta = np.dot(p_theta,rst)
        sum_theta = sum_theta + np.linalg.norm(delta_theta)**2
        #print(delta_theta)
        #print('norm',np.linalg.norm(delta_theta))

        #sum_theta = sum_theta + np.linalg.norm(delta_theta,2)**2
        #print(sum_theta)
        #print(np.linalg.norm(delta_theta,2)**2)
    #print('sum_theta/resultsNum:',sum_theta/len(results))
    rms_value = np.sqrt(sum_theta/len(results))

    testsum = 0.0
    for rst in results:
        testsum = testsum + np.linalg.norm(rst - true_val)
    print('norm distance:',testsum/len(results))
    #print(rms_value)
    return rms_value

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

    #print('weight:',weight)

    M = createMMat(data, weight)
    #M = np.linalg.inv(M)
    #M = np.linalg.inv(M)
    lamda,v = np.linalg.eigh(M)

    sorted_lamda = sorted(lamda, reverse=True)
    print('sorted_lamda',sorted_lamda)
    sorted_lamda = np.matrix(sorted_lamda[:5]).T

    sum_inv_ramda = 0.0
    #sum_inv_ramda = sorted_lamda.sum()
    for i in range(sorted_lamda.shape[0]):
        sum_inv_ramda = sum_inv_ramda + (1.0/sorted_lamda[i])
    #print('M:',M)
    #print(createMMat(data, np.matrix(np.ones(data.shape[0])).T))
    #sum_inv_ramda = 0.0
    #for i in range(lamda.shape[0]):
        #sum_inv_ramda = sum_inv_ramda + 1.0/lamda[i]

    Dkcr = stdv*np.sqrt(sum_inv_ramda) / np.sqrt(data.shape[0])
    #array_info(lamda)
    #inv_M = np.linalg.inv(M)
    #print(inv_M)
    #print(M*inv_M)
    #diag_M = np.diag(inv_M).sum()
    #print(diag_M)
    #print(stdv/np.sqrt(data.shape[0]))
    #print(M*true)
    #la,v = np.linalg.eigh(M)
    #print(la.sum())
    #Dkcr = ( stdv/np.sqrt(data_num) )*np.sqrt(diag_M)
    #Dkcr = ( stdv/np.sqrt(data.shape[0]) )*np.sqrt(diag_M)
    return Dkcr[0][0]

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

    #Plot last answer figure of Ellipse
    #plotData(myuLSM,myuFNS,trueAns,dataNoised)

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

def calcRepLSMandMSE():
    #read data of points including error value
    data = np.loadtxt('points.dat',comments='!')
    #read data of true value
    trueAns = np.loadtxt('true_param.dat')
    #init value
    trialNum = 1000
    #List of standard deviation
    stdvList = np.arange(0.0, 0.101, 0.005, dtype = 'float64')
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


    #x = range(4, 4, 100)
    #print('x',x)
    #for i in x:
        #i = i * 0.1
