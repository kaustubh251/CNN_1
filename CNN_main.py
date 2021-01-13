import numpy as npy
from scipy import signal as sig

file = open('ex3data1X.csv','rt')
dataX = file.read()
dataXsplit = dataX.split()
dataX = []
for i in range(5000):
    dataX_row = dataXsplit[i].split(',')
    dataX_rowFloat = []
    for j in range(400):
        dataX_rowFloat.append(float(dataX_row[j]))
    dataX.insert(i, dataX_rowFloat)

file = open('ex3data1y.csv', 'rt')
dataY = file.read()
dataYsplit = dataY.split()
dataY = []
for element in range(len(dataYsplit)):
    dataY.append(int(dataYsplit[element]))

def sigmoid(z):
    return (1/(1 + npy.exp(-z)))

def trainingExampleProvider(x, y, dataX, dataY):
    x1 = []
    for i in range(20):
        x1_row = []
        for j in range(20):
            x1_row.append(dataX[x+i][y+j])
        x1.insert(i, x1_row)
    y1 = npy.zeros([10,1])
    num = dataY[20*x + y]
    if num == 10:
        y1[0] = 1
    else:
        y1[num] = 1
    a, b = npy.meshgrid(npy.linspace(-1,1,3), npy.linspace(-1,1,3))
    d = npy.sqrt(a*a+b*b)
    sigma, mu = 1.5, 0.0
    g = npy.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    x1_conv = sig.convolve2d(x1, g, 'valid')
    i = int(len(x1_conv)/3)
    j = int(len(x1_conv[0])/3)
    avg = (1/9)*npy.ones([3,3])
    x1_avg = []
    for x in range(i):
        x1_avgRow = []
        for y in range(j):
            x1_convSect = x1_conv[3*x:(3*x+3),3*y:(3*y+3)]
            mat1 = sig.convolve2d(x1_convSect, avg, 'valid')
            num1 = mat1[0][0]
            x1_avgRow.append(num1)
        x1_avg.insert(x, x1_avgRow)
    x1_out = []
    for i in range(len(x1_avg)):
        x1_outRow = []
        for j in range(len(x1_avg[0])):
            x1_outRow.append(sigmoid(x1_avg[i][j]))
        x1_out.insert(i, x1_outRow)
    x1_vector = [1]
    for i in range(len(x1_out)):
        for j in range(len(x1_out[0])):
            x1_vector.append(x1_out[i][j])
    return x1_vector, y1

#x = 0
#y = 0
#x1,y1 = trainingExampleProvider(x,y,dataX,dataY)
#print(y1)

def outputBlackBox(x1, theta1, theta2):
    a2 = [1]
    for i in range(len(theta1)):
        z = 0
        for j in range(len(theta1[0])):
            z += theta1[i][j]*x1[j]
        a2.append(sigmoid(z))
    a3 = []
    for i in range(len(theta2)):
        z = 0
        for j in range(len(theta2[0])):
            z += theta2[i][j]*a2[j]
        a3.append(sigmoid(z))
    return a2, a3

def cost(dataX, dataY, theta1, theta2, regParam):
    m = int(len(dataX)/20)
    n = int(len(dataX[0])/20 - 5)
    cost1 = 0
    for x in range(m):
        for y in range(n):
            x1, y1 = trainingExampleProvider(x, y, dataX, dataY)
            a2, a3 = outputBlackBox(x1, theta1, theta2)
            for z in range(len(a3)):
                cost1 += -(y1[z]*npy.log(a3[z]) + (1 - y1[z])*npy.log(1 - a3[z]))
    cost2 = 0
    for i in range(len(theta1)):
        for j in range(len(theta1[0])):
            cost2 += theta1[i][j]*theta1[i][j]
    for i in range(len(theta2)):
        for j in range(len(theta2[0])):
            cost2 += theta2[i][j]*theta2[i][j]
    cost2 = cost2*regParam/2
    cost = cost1 + cost2
    cost = cost/(m*n)
    return cost

def backPropagation(dataX, dataY, theta1, theta2, regParam, alpha, maxIter):
    m = int(len(dataX)/20)
    n = int(len(dataX[0])/20 - 5)
    for count in range(maxIter):
        grad2 = npy.zeros([len(theta2), len(theta2[0])])
        grad1 = npy.zeros([len(theta1), len(theta1[0])])
        for x in range(m):
            for y in range(n):
                x1, y1 = trainingExampleProvider(x, y, dataX, dataY)
                a2, a3 = outputBlackBox(x1, theta1, theta2)
                del3 = []
                for c1 in range(len(a3)):
                    del3.append(a3[c1] - y1[c1])

                del2 = []
                for c1 in range(len(theta2[0])):#101
                    del2_row = 0
                    for c2 in range(len(theta2)): #10
                        del2_row += del3[c2]*theta2[c2][c1]
                        grad2[c2][c1] += a2[c1]*del3[c2]
                    del2.append(del2_row*a2[c1]*(1-a2[c1]))
                for c1 in range(len(theta1)): #100
                    for c2 in range(len(theta1[0])): #401
                        grad1[c1][c2] += x1[c2]*del2[c1]
        for i in range(len(theta2)):
            for j in range(len(theta2[0])):
                if j==0:
                    grad2[i][j] = grad2[i][j]/(m*n)
                else:
                    grad2[i][j] = grad2[i][j]/(m*n) + regParam*theta2[i][j]
        for i in range(len(theta1)):
            for j in range(len(theta1[0])):
                if j==0:
                    grad1[i][j] = grad1[i][j]/(m*n)
                else:
                    grad1[i][j] = grad1[i][j]/(m*n) + regParam*theta1[i][j]
        for i in range(len(theta2)):
            for j in range(len(theta2[0])):
                theta2[i][j] = theta2[i][j] - alpha*grad2[i][j]
        for i in range(len(theta1)):
            for j in range(len(theta1[0])):
                theta1[i][j] = theta1[i][j] - alpha*grad1[i][j]
        print(theta1)
        print(theta2)
    return theta1, theta2

theta1_initial = npy.random.rand(18, 37)
theta2_initial = npy.random.rand(10, 19)
regParam = 10
alpha = 0.03
maxIter = 10
theta1, theta2 = backPropagation(dataX, dataY, theta1_initial, theta2_initial, regParam, alpha, maxIter)
p = 0
for x in range(int(len(dataX)/20)):
    for y in range(int(len(dataX[0])/20 - 15)):
        x1, y1 = trainingExampleProvider(x, (y+15), dataX, dataY)
        a2, a3 = outputBlackBox(x1, theta1, theta2)
        max = a3[0]
        maxIndex = 0
        for i in range(len(a3)):
            if a3[i]>max:
                max = a3[i]
                maxIndex = i
        if int(y1[maxIndex]) == 1:
            p += 1
accuracy = p*100/((len(dataX)/20)*(len(dataX[0])/20 - 15))
print(accuracy)