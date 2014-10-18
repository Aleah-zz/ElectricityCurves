import sys, math, random
import fileinput
import string
from string import whitespace
import re
import time
import csv
import pylab as pl

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
        self.price = 0
        self.volume = 0

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

def kmeans(points, k, cutoff, distanceFunc, initMethod = "Heuristic"):
   
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
        
        if biggest_shift < cutoff:
            break
        Iter = Iter + 1

    return clusters

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
            subst = matrix[i][j] + abs(seq1[i][1] - seq2[j][1])

            matrix[i+1][j+1] = min (delete,insert,subst)
    
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
        print chain1
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

def vizualizationClusters(clusters, pr, vol):

    numberOfClusters = len(clusters)
    colors = ['r', 'b', 'g', 'c', 'w']


    for i in range(numberOfClusters):
        x = []
        y = []
        for j in range(clusters[i].getLength()):
            dateAndHour = str(clusters[i].getElement(j))
            x.append(pr[dateAndHour])
            y.append(vol[dateAndHour])
        print len(x),len(y)            
        fit = pl.polyfit(x, y, 1)
        fit_fn = pl.poly1d(fit)
        
        pl.plot(x,y, 'yo', x, fit_fn(x), '--k', color = colors[i])
        pl.savefig("./pic/test"+str(i)+".png")


def main():


    # load train set
    chainsTrainSet = loadData("./data/dataTrain1half.dt")
    
    # load test set
    chainsValidationSet = loadData("./data/dataValidation1half.dt")

    # load price
    prices = loadAdditionalData("./data/2012price.csv")
    
    # load volume
    volumes = loadAdditionalData("./data/2012volume.csv")

    # Make matrix of distnaces and load it 
    # filename = "Matrix.py"
    # makeMatrix(filename, chainsTrainSet)
    global matrixDist
    from Matrix import matrixDist

    clNum = [5] # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    
    # open file to write results of clustering
    CSVfileName = "./results/temp.csv"
    f = open(CSVfileName, "w")
    f.write("numberOfClusters;Cluster;Date;Hour;isValidation;isCentroid\n")  #write header of CSV file

    for numberOfClusters in clNum:

        clusters = kmeans(chainsTrainSet, numberOfClusters, 1000, LevenshteinDistance, "Heuristic") #
        validation(chainsValidationSet, clusters)
        vizualizationClusters(clusters,prices,volumes)
        # writeToCSV(f, clusters)
        # print clusters

    f.close()



if __name__ == "__main__":
    main()
