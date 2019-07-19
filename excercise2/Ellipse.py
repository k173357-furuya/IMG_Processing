# -*- coding: utf-8 -*-

from sympy import *
import sympy
import numpy as np
import matplotlib.pyplot as plt
import math

def generateVecFromEllipse(axis, center, T, rng=[0, 2*math.pi], num=101):

  t = np.linspace(rng[0], rng[1], num)
  t = np.reshape(t, (t.shape[0], 1))

  xVec = np.zeros((t.shape))
  yVec = np.zeros((t.shape))
  for i in range(t.shape[0]):
    xVec[i] = axis[0, 0] * math.cos(t[i])
    yVec[i] = axis[1, 0] * math.sin(t[i])

  dataTmp = np.concatenate((xVec, yVec),  axis=1)
  dataTmp = T * dataTmp.T
  dataTmp = dataTmp.T
  dataTmp[:, 0] =  dataTmp[:, 0] + center[0, 0]
  dataTmp[:, 1] =  dataTmp[:, 1] + center[1, 0]

  return dataTmp

def generateEllipseEqn(axis, center, T):

  # 1. Creating Vector with Symbol
  x_ = Symbol('x_')
  y_ = Symbol('y_')
  x_vec = np.matrix([[x_ - center[0, 0]], [y_ - center[1, 0]]])

  # 2. Define Original Ellipse Matrix
  # No tilt
  ElMat = np.matrix([[1/axis[0, 0]**2, 0], [0, 1/axis[1, 0]**2]])

  # 3. Rotate Coordinate
  Tx_ = T.T * x_vec

  # 4. Expand Ellipse Eqn.
  fn = sympy.expand((Tx_.T * (ElMat * Tx_))[0, 0] - 1)

  A = float(fn.coeff(x_, 2))
  B = float(fn.coeff(x_*y_, 1))
  C = float(fn.coeff(y_, 2))
  D = float(fn.subs([(y_, 0)]).coeff(x_, 1))
  E = float(fn.subs([(x_, 0)]).coeff(y_, 1))
  F = float(fn.subs([(x_, 0), (y_, 0)]))

  print("The ellipse eqn you are looking for is :")
  print(fn)
  return A, B, C, D, E, F

def getEllipseProperty(A, B, C, D, E, F):

  if A < 0:
    A = -A
    B = -B
    C = -C
    D = -D
    E = -E
    F = -F

  # Spectral Decomposition
  M = np.matrix([[A, B/2], [B/2, C]])
  lamdas, v = np.linalg.eigh(M)

  # Diagonalize Coeffs Matrix
  DiagA = v.T * M * v

  # Apply translation for 1st order term.
  tmp = np.matrix([D, E]) * v

  # Calculate coefficient in rotated coords
  AA = DiagA[0, 0]
  BA = DiagA[0, 1] + DiagA[1, 0]
  CA = DiagA[1, 1]
  DA = tmp[0, 0]
  EA = tmp[0, 1]
  scale = F - DA**2 / (4*AA) - EA**2 / (4*CA)

  # Normalize coeffs wrt to constant term.
  AA = AA / abs(scale)
  BA = BA / abs(scale)
  CA = CA / abs(scale)
  DA = DA / abs(scale)
  EA = EA / abs(scale)
  FA = abs(scale) / abs(scale)

  # Ellipse Property Extraction
  a = 1 / math.sqrt(abs(lamdas[0] / scale))
  b = 1 / math.sqrt(abs(lamdas[1] / scale))

  T = np.matrix([[v[0, 0], v[0, 1]], [v[1, 0], v[1, 1]]])
  trans = v.T * np.matrix([[-DA/(2*AA)], [-EA/(2*CA)]])

  valid = True
  if AA * CA < 0:
    valid = False

  return valid, np.matrix([[a], [b]]), trans, T

def plotData(dataOrg, dataEst):

  fig, ax = plt.subplots(ncols = 1, figsize=(10, 10))
  plt.xlim(-10, 10)
  plt.ylim(-10, 10)
  ax.plot(dataOrg[:, 0], dataOrg[:, 1])
  ax.plot(dataEst[:, 0], dataEst[:, 1])

if __name__ == "__main__":

  print("Ellipse Drawing Sample")

  # Define ellipse property.
  axisXOrg = 3
  axisYOrg = 1
  centerX = 5
  centerY = 3
  rad = math.radians(120)

  # Prepare arguments.
  axisOrg = np.matrix([[axisXOrg], [axisYOrg]])
  Rorg = np.matrix([[cos(rad), -sin(rad)], [sin(rad), cos(rad)]])
  centOrg = np.matrix([[centerX], [centerY]])

  # 0. Generating Data and Elilpse Equation Coefficient
  dataOrg = generateVecFromEllipse(axisOrg, centOrg, Rorg)
  A, B, C, D, E, F = generateEllipseEqn(axisOrg, centOrg, Rorg)

  # 1. Extract Elilpse Property from Egn
  valid, axis, centerEst, Rest = getEllipseProperty(A, B, C, D, E, F)

  # 2. Generating Ellipse Point and Rotate
  dataEst = generateVecFromEllipse(axis, centerEst, Rest)

  # 3. Plot Data
  plotData(dataOrg, dataEst)
