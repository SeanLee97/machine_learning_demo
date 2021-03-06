# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
绘制决策树
代码整理于网络，出自《机器学习实战》
'''
import matplotlib.pyplot as plt

#定义文本框和箭头格式  
decisionNode = dict(boxstyle="sawtooth", fc="0.8") #定义判断节点形态  
leafNode = dict(boxstyle="round4", fc="0.8") #定义叶节点形态  
arrow_args = dict(arrowstyle="<-") #定义箭头  
  
#绘制带箭头的注解  
#nodeTxt：节点的文字标注, centerPt：节点中心位置,  
#parentPt：箭头起点位置（上一节点位置）, nodeType：节点属性  
def plotNode(nodeTxt, centerPt, parentPt, nodeType):  
    draw.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',  
             xytext=centerPt, textcoords='axes fraction',  
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args ) 
#计算叶节点数  
def getNumLeafs(myTree):  
    numLeafs = 0  
    firstStr = list(myTree.keys())[0]   
    secondDict = myTree[firstStr]   
    for key in secondDict.keys():  
        if type(secondDict[key]).__name__=='dict':#是否是字典  
            numLeafs += getNumLeafs(secondDict[key]) #递归调用getNumLeafs  
        else:   numLeafs +=1 #如果是叶节点，则叶节点+1  
    return numLeafs  
  
#计算数的层数  
def getTreeDepth(myTree):  
    maxDepth = 0  
    firstStr = list(myTree.keys())[0]  
    secondDict = myTree[firstStr]  
    for key in secondDict.keys():  
        if type(secondDict[key]).__name__=='dict':#是否是字典  
            thisDepth = 1 + getTreeDepth(secondDict[key]) #如果是字典，则层数加1，再递归调用getTreeDepth  
        else:   thisDepth = 1  
        #得到最大层数  
        if thisDepth > maxDepth:  
            maxDepth = thisDepth  
    return maxDepth  

#在父子节点间填充文本信息  
#cntrPt:子节点位置, parentPt：父节点位置, txtString：标注内容  
def plotMidText(cntrPt, parentPt, txtString):  
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]  
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]  
    draw.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30) 


#绘制树形图  
#myTree：树的字典, parentPt:父节点, nodeTxt：节点的文字标注  
def plotTree(myTree, parentPt, nodeTxt):  
    numLeafs = getNumLeafs(myTree)  #树叶节点数  
    depth = getTreeDepth(myTree)    #树的层数  
    firstStr = list(myTree.keys())[0]     #节点标签  
    #计算当前节点的位置  
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  
    plotMidText(cntrPt, parentPt, nodeTxt) #在父子节点间填充文本信息  
    plotNode(firstStr, cntrPt, parentPt, decisionNode) #绘制带箭头的注解  
    secondDict = myTree[firstStr]  
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  
    for key in secondDict.keys():  
        if type(secondDict[key]).__name__=='dict':#判断是不是字典，  
            plotTree(secondDict[key],cntrPt,str(key))        #递归绘制树形图  
        else:   #如果是叶节点  
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)  
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD  
  
#创建绘图区  
def draw(inTree):  
    fig = plt.figure(1, facecolor='white')  
    fig.clf()  
    axprops = dict(xticks=[], yticks=[])  
    draw.ax1 = plt.subplot(111, frameon=False, **axprops)      
    plotTree.totalW = float(getNumLeafs(inTree)) #树的宽度  
    plotTree.totalD = float(getTreeDepth(inTree)) #树的深度  
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;  
    plotTree(inTree, (0.5,1.0), '')  
    plt.show()  