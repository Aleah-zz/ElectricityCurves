import sys, math, random
import fileinput
import string
from string import whitespace
import re
import os
import csv
import pylab as pl
import datetime
import operator

global matrixDist
matrixDist = {};

class Chain:
    # Class represent chain (sequence of facts with weights)
    def __init__(self, elements, idP='cent', isTest = 0, reference=None):
        self.elements = elements
        self.idP = idP
        self.n = len(elements)
        self.reference = reference
        self.isTest = isTest
        
    def __repr__(self):
        return str(self.idP) # + ' - ' + str(self.elements)

    def printChainId(self):
        return str(self.idP)

class Cluster:
    # Class represent clusters
    def __init__(self, points, distanceFunc):
        
        self.points = points

        if len(points) > 0:
            self.centroid = self.calculateCentroid(distanceFunc)

    def __repr__(self):
        return str(self.points)

    def getLength(self):
        return len(self.points)

    def getElement(self, ID):
        return self.points[ID].printChainId()

    def getCentroid(self):
        return self.centroid

    def update(self, points, distanceFunc):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid(distanceFunc)
        return distanceFunc(old_centroid, self.centroid)

    def calculateCentroid(self,distanceFunc):

        minDist =10000000
        for p1 in self.points:
            dist = 0
            for p2 in self.points:
                dist = dist + distanceFunc(p1,p2)
            if minDist>dist:
                minDist = dist
                centroid = p1
        return centroid

def kmeans(points, k, cutoff, distanceFunc, initMethod = "Heuristic", prices = [], volumes = [], vizEachIteration = False):
   
    # Initialisation. It depends on method we choose to initialize: Heuristic or Random
    if (initMethod=="Heuristic"):
        initial = []
        while len(initial)<k:
            rand = random.sample(points, 1)[0]
            alreadyIn = False
            for i in initial:
                if (i.n == rand.n):
                    alreadyIn = True
            if not alreadyIn:
                initial.append(rand)
    else:
        initial = random.sample(points, k)
    

    clusters = [Cluster([p],distanceFunc) for p in initial]
    Iter = 0
    
    while Iter<15:
        
        AccSum = 0
        lists = [ [] for c in clusters]
        for p in points:
            smallest_distance = distanceFunc(p,clusters[0].centroid)
            index = 0
            for i in range(len(clusters[1:])):
                distance = distanceFunc(p, clusters[i+1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i+1
            AccSum = AccSum + smallest_distance
            lists[index].append(p)

        biggest_shift = 0.0

        numCl = len(clusters)

        for i in range(numCl):
            if len(lists[i])==0:
                del clusters[i]
        
        for i in range(len(clusters)):
            shift = clusters[i].update(lists[i],distanceFunc)
            biggest_shift = max(biggest_shift, shift)
        
        if (vizEachIteration):
            vizualizationClusters(clusters, prices, volumes, Iter)

        if biggest_shift < cutoff:
            break

        Iter = Iter + 1


        # print biggest_shift
    return clusters, Iter

def LD (a,b):
   
    seq1 = a.elements
    seq2 = b.elements
    
    matrix = [0] * (len(seq1)+1)
    for i in xrange (len(seq1)+1):
        matrix[i] = [0] * (len(seq2)+1)
    
    for j in xrange (len(seq2)):
        matrix[0][j+1] = matrix[0][j] + seq2[j][1]
        
    for i in xrange(len(seq1)):
        matrix[i+1][0] = matrix[i][0] + seq1[i][1]
        
        for j in xrange(len(seq2)):
            delete =  matrix[i][j+1] + seq1[i][1]
            insert = matrix[i+1][j] + seq2[j][1]
            # previouos simple version
            # subst = matrix[i][j] + abs(seq1[i][1] - seq2[j][1])

            if (seq1[i][0] == seq2[j][0]):
                subst = matrix[i][j] + abs(seq1[i][1] - seq2[j][1])
            else:
                subst = matrix[i][j] + abs(seq1[i][1] + seq2[j][1])

            matrix[i+1][j+1] = min(delete,insert,subst)
    
    return matrix[len(seq1)][len(seq2)]
    # return thisrow[len(seq2) - 1]

def LevenshteinDistance (a,b):

    return matrixDist[str(a)][str(b)]

def loadData(fileName, isTest = 0):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        tempChain = []
        chains = []
        for row in reader:
            if (float(row[3]) == 0):
                if len(tempChain)>0:
                    chains.append(Chain(tempChain,idCh, isTest))
                tempChain = []
            element = []
            element.append(float(row[3]))
            element.append(float(row[4]))
            tempChain.append(element)
            idCh = row[0] +";"+ row[1]
            
        if len(tempChain)>0:
            chains.append(Chain(tempChain,idCh, isTest))

    return chains

def loadAdditionalData(fileName):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        data = {}
        next(reader)

        for row in reader:
            dateAndHour = row[0] +";"+ row[1]
            data[dateAndHour] = float(row[2])

    return data

def makeMatrix(filename, chains):

    f = open("./" + filename, "w")

    f.write("matrixDist ={} \n")
           
    global matrixDist
    # print
    for chain1 in chains:
        matrixDist[chain1] = {}
        f.write("matrixDist['"+ str(chain1) +"']={} \n")
        
        for chain2 in chains:
            if (chain2 in matrixDist) and (chain1 in matrixDist[chain2]):
                matrixDist[chain1][chain2] = matrixDist[chain2][chain1]
            else:
                matrixDist[chain1][chain2] = LD(chain1,chain2)
    
            f.write("matrixDist['"+ str(chain1) +"']['" + str(chain2) + "']="+ str(matrixDist[chain1][chain2]) + " \n")  
            
    f.close()

def validation(chains, clusters):

    result = {}

    for nCh,chain in enumerate(chains):
        minDist = 100000000
        for nCl,cluster in enumerate(clusters):
            d = LD(cluster.centroid,chain)
            if minDist>d:
                minDist = d
                result[nCh] = nCl
        clusters[result[nCh]].points.append(chains[nCh])

def writeToCSV (fileCSV, clusters):

    numberOfClusters = len(clusters)

    for i in range(numberOfClusters):
        for j in range(clusters[i].getLength()):
            if (str(clusters[i].getElement(j))==str(clusters[i].centroid)):
                isCentr = 1
            else:
                isCentr = 0

            fileCSV.write(str(numberOfClusters)+ ';'+ str(i) + ';' + str(clusters[i].getElement(j)) + ';' + str(clusters[i].points[j].isTest) +";"+ str(isCentr) + "\n")  

def evaluationOfR2(x_mass, linModel, y_mass):
    R2 = 0
    
    for num,x in enumerate(x_mass):
        y = y_mass[num]
        R2 = R2 + (y - linModel(x))**2
    return R2

def vizualizationClusters(clusters, pr, vol, Name=0, picFormat = "png", withLabels = False):

    numberOfClusters = len(clusters)
    colors = ['r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w''r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b']
    
    fig = pl.figure()
    
    R2 = {}
    R2all = 0
    labels = []
    xall =[]
    yall = []
    
    for i in range(numberOfClusters):
        xNoTest = []
        yNoTest = []
        xTest = []
        yTest = []
        for j in range(clusters[i].getLength()):
            dateAndHour = str(clusters[i].getElement(j))
            labels.append(dateAndHour)
            if (clusters[i].points[j].isTest==0):
                xNoTest.append(vol[dateAndHour])
                yNoTest.append(pr[dateAndHour])
            else:
                xTest.append(vol[dateAndHour])
                yTest.append(pr[dateAndHour])

        fit = pl.polyfit(xNoTest, yNoTest, 1)
        fit_fn = pl.poly1d(fit)
        
        R2[i] = evaluationOfR2(xTest,fit_fn,yTest)
        R2all = R2all + R2[i] 

        pl.plot(xNoTest,yNoTest, 'yo', xNoTest, fit_fn(xNoTest), '--k', color = colors[i])
        pl.plot(xTest,yTest, 'y*', color = colors[i])
        pl.xlabel("Volume, MWh")
        pl.ylabel("Price, Rubles per MWh")

        xall = xall + xNoTest + xTest
        yall = yall + yNoTest + yTest
    

    # annotation
    if withLabels:
        for i, lab in enumerate(labels):
            pl.annotate(lab, xy = (xall[i], yall[i]), xytext = (-5, 5), textcoords = 'offset points', ha = 'right', va = 'bottom' , fontsize = 5)

    pl.savefig("./pic/"+str(pathToPic) + "/" + str(Name)+"test"+str(i)+"."+picFormat, format=picFormat)
    pl.close(fig)

    return R2all

def inClusterOutliersDetection(clusters):
    distnaces = {}
    distMax = 0
    for Ncl,cl in enumerate(clusters):
        distnaces[Ncl] = {}
        for ch1ID in cl.points:
            distSumm = 0
            for ch2ID in cl.points:
                distSumm = distSumm + matrixDist[str(ch1ID)][str(ch2ID)]
            distnaces[Ncl][ch1ID] = distSumm
        distnaces[Ncl] = sorted(distnaces[Ncl].items(), key=operator.itemgetter(1), reverse=True)
    return distnaces   

def outliersDetection():
    # not used
    distnaces = {}
    distMax = 0
    for ch1ID in matrixDist:
        distSumm = 0
        for ch2ID in matrixDist[ch1ID]:
            distSumm = distSumm + matrixDist[ch1ID][ch2ID]
        distnaces[ch1ID] = distSumm
    return distnaces     

def vizualizationCurves (clusters, picFormat = "png"):

    colors = ['r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w''r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b', 'g', 'c', 'k', 'm' ,'y','w', 'r', 'b']
    fig = pl.figure()
    for Ncl,cl in enumerate(clusters):
           
        for curves in cl.points:
            x, y = [], []
            sumX, sumY = 0,0
            for el in curves.elements:
                x.append(sumX + el[1])
                sumX = sumX + el[1]
                y.append(el[0])
            pl.step(x,y, color = colors[Ncl])


    # pl.xlim(50000, 130000)
    # pl.ylim(0, 5000)
    pl.xlabel("Volume, MWh")    
    pl.ylabel("Price, Rubles per MWh")
    
    pl.savefig("./pic/"+str(pathToPic) + "/curvesFromCluster." + picFormat, format=picFormat)
    pl.close(fig)

def QualityIndexes(clusters):
    
    # cluster diam
    diam, currentdiam = 0, {}
    diamCl, diami, diamj = 0,0,0
    
    SSw = 0
    for Ncl,cl in enumerate(clusters):
        currentdiam[Ncl] = 0
        for ci in cl.points:
            for cj in cl.points:
                SSw = SSw + matrixDist[str(ci)][str(cj)]*matrixDist[str(ci)][str(cj)]
                if matrixDist[str(ci)][str(cj)] > currentdiam[Ncl]:
                    currentdiam[Ncl] = matrixDist[str(ci)][str(cj)]
                    
        if currentdiam[Ncl]>diam:
            diam = currentdiam[Ncl]
    
    # between clusters distance
    between = 10000000
    betweenCl1, betweenCl2, betweeni, betweenj = 0,0,0,0
    
    SSb = 0
    for Ncl1,cl1 in enumerate(clusters):
        for Ncl2,cl2 in enumerate(clusters):
            if Ncl2!=Ncl1:
                for ci in cl1.points:
                    for cj in cl2.points:
                        SSb = SSb + matrixDist[str(ci)][str(cj)]*matrixDist[str(ci)][str(cj)]
                        if matrixDist[str(ci)][str(cj)] < between:
                            between = matrixDist[str(ci)][str(cj)]
                            betweenCl1, betweenCl2, betweeni, betweenj = Ncl1,Ncl2,ci,cj
                            
    # print SSb/1000000000, SSw/1000000000, (SSb+SSw)/1000000000
    return between/diam, SSb/(SSw+SSb)
    
def main():


    # load train set
    chainsTrainSet = loadData("./Data/subDataTrain1half.dt")
    
    # load test set
    chainsValidationSet = loadData("./Data/dataValidation1half.dt", isTest = 1)

    # load price
    prices = loadAdditionalData("./Data/2012price.csv")
    
    # load volume
    volumes = loadAdditionalData("./Data/2012volume.csv")

    # Make matrix of distnaces and load it 
    # filename = "Matrix1HalfExp.py"
    # makeMatrix(filename, chainsTrainSet)
    
    global matrixDist
    from Matrix1HalfExp import matrixDist

    # outlier detection
    # top 10 points which can be outliers
    # tempdistances = outliersDetection()
    # outliers = sorted(tempdistances.items(), key=operator.itemgetter(1), reverse=True)
    # print outliers
        
    clNum = [6] #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    
    # Creating directory for pictures 
    d = datetime.datetime.today()
    global pathToPic

    pathToPic = d.strftime("%y%m%d_%H%M%S")
    os.mkdir("./pic/"+str(pathToPic))       

    # open file to write results of clustering
    CSVfileName = "./Results/temp.csv"
    f = open(CSVfileName, "w")
    f.write("numberOfClusters;Cluster;Date;Hour;isValidation;isCentroid\n")  #write header of CSV file

    print len(chainsTrainSet)
    R2all = {}

    for numberOfClusters in clNum:
        clusters, numberOfIterations = kmeans(chainsTrainSet, numberOfClusters, 0.1, LevenshteinDistance, "Heuristic", prices, volumes, vizEachIteration = False) #
        out = inClusterOutliersDetection(clusters)
        # validation(chainsValidationSet, clusters)
        R2all[numberOfClusters] = vizualizationClusters(clusters,prices,volumes,numberOfClusters, "png", withLabels = True)
        vizualizationCurves(clusters, "png")
        writeToCSV(f, clusters)
        # DI, RSIndex = QualityIndexes(clusters) #not work with validation, because there no validation points in MatrixFile
        print numberOfClusters,R2all[numberOfClusters]

    print out 

    f.close()

if __name__ == "__main__":
    main()
