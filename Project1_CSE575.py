import numpy
import scipy.io
import math
import geneNewData
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score

def main():
    myID='9328' #your ID here
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
   
    print('Your trainset and testset are generated successfully!')
 
    
    #Task 1 &  # Task 2  
    #Create Average and stdDev of TrainSet0 and TrainSet1  
    
    NewArrayTrainSet0 = numpy.zeros(5000,dtype = float)
    StdDevBright0 = numpy.zeros(5000,dtype = float)
        
    #Create Average and stdDev of TrainSet1
    NewArrayTrainSet1 = numpy.zeros(5000, dtype = float)
    stdDevArray1 = numpy.zeros(5000, dtype = float)
    
    #Create Average and stdDev of TestSet0
    NewArrayTestSet0 = numpy.zeros(980, dtype = float )
    stdDevArray2 = numpy.zeros(980, dtype = float)
        
    
    #Create Average and stdDev of TestSet1  
    NewArrayTestSet1 = numpy.zeros(1135, dtype = float)
    stdDevArray3 = numpy.zeros(1135, dtype = float)
    

    for j in range(5000):
    #Calculate Mean/Std Dev                                    

        NewArrayTrainSet0[j]     = train0[j].mean(dtype = float)
        StdDevBright0[j]         = train0[j].std(dtype = float)
                                                                     
        NewArrayTrainSet1[j] = train1[j].mean(dtype = float)         
        stdDevArray1[j]      = train1[j].std(dtype = float)
    
   
    #TrainSet0 
    MeanOfAverageBrightness  = (NewArrayTrainSet0.sum() /5000)  
    VarOfAverageBrightess    = NewArrayTrainSet0.var()
    MeanOfStdDevBrightness   = (StdDevBright0.sum() / 5000) 
    VarOfStdDevBrightness    = StdDevBright0.var() 
    
    #TrainSet1
    MeanOfAverageBrightness1  = (NewArrayTrainSet1.sum() /5000)  
    VarOfAverageBrightess1    = NewArrayTrainSet1.var() 
    MeanOfStdDevBrightness1   = (stdDevArray1.sum() / 5000) 
    VarOfStdDevBrightness1    = stdDevArray1.var()

    
    
    #printTrainSetValues
    print(MeanOfAverageBrightness)
    print(VarOfAverageBrightess)
    print(MeanOfStdDevBrightness)
    print(VarOfStdDevBrightness)
    print(MeanOfAverageBrightness1)
    print(VarOfAverageBrightess1)
    print(VarOfStdDevBrightness1)
   # I need to get the mean / standard deviation for each test1 image                       
   
    for i in range(980):
   #I need to store my Average Pixel Brightness/Std of test0                              - 
        NewArrayTestSet0[i]  = test0[i].mean(dtype = float) 
        stdDevArray2[i]      = test0[i].std(dtype = float) 
   
    
   #I need to get the mean/variation deviation of test0                                      
    NewArrayTestSet0Result     = NewArrayTestSet0.sum()/980
    NewArrayTestSet0ResultVar  = NewArrayTestSet0.var()
    stdDevArray2Result         = stdDevArray2.sum()/980
    stdDevArray2ResultVar      = stdDevArray2.var()
    
    #Print Values of TestSet0
    print('TestSet0')
    print(NewArrayTestSet0Result)
    print(NewArrayTestSet0ResultVar)
    print(stdDevArray2Result)
    print(stdDevArray2ResultVar)

   #I need to store my Average Pixel Brightness/Std of test1                             
    for x in range(1135):
   #  I need to store my Average Pixel Brightness of test2                              
        NewArrayTestSet1[x]  = test1[x].mean(dtype = float)
        stdDevArray3[x]      = test1[x].std(dtype  = float)
        
   #I need to get the mean/variation deviation of test1                                    
        
    MeanOfAverageBrightnessTest1  = NewArrayTestSet1.sum() /1135 
    VarOfAverageBrightessTest1    = NewArrayTestSet1.var()
    MeanOfStdDevBrightnessTest1   = stdDevArray3.sum() / 1135 
    VarOfStdDevBrightness1        = stdDevArray3.var()              
    
    print('TestSet1')
    print(MeanOfAverageBrightnessTest1)
    print(VarOfAverageBrightessTest1)
    print(MeanOfStdDevBrightnessTest1)
    print(VarOfStdDevBrightness1)
   
     
   
      
    
    import numpy
import scipy.io
import math
import geneNewData


def main():
    myID='9328' #your ID here
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
   
    print('Your trainset and testset are generated successfully!')
 
    
    #Task 1 &  # Task 2  
    #Create Average and stdDev of TrainSet0 and TrainSet1  
    
    NewArrayTrainSet0 = numpy.zeros(5000,dtype = float)
    StdDevBright0 = numpy.zeros(5000,dtype = float)
        
    #Create Average and stdDev of TrainSet1
    NewArrayTrainSet1 = numpy.zeros(5000, dtype = float)
    stdDevArray1 = numpy.zeros(5000, dtype = float)
    
    #Create Average and stdDev of TestSet0
    NewArrayTestSet0 = numpy.zeros(980, dtype = float )
    stdDevArray2 = numpy.zeros(980, dtype = float)
        
    
    #Create Average and stdDev of TestSet1  
    NewArrayTestSet1 = numpy.zeros(1135, dtype = float)
    stdDevArray3 = numpy.zeros(1135, dtype = float)
    

    for j in range(5000):
    #Calculate Mean/Std Dev                                    

        NewArrayTrainSet0[j]     = train0[j].mean(dtype = float)
        StdDevBright0[j]         = train0[j].std(dtype = float)
                                                                     
        NewArrayTrainSet1[j] = train1[j].mean(dtype = float)         
        stdDevArray1[j]      = train1[j].std(dtype = float)
    
   
    #TrainSet0 
    MeanOfAverageBrightness  = (NewArrayTrainSet0.sum() /5000)  
    VarOfAverageBrightess    = NewArrayTrainSet0.var()
    MeanOfStdDevBrightness   = (StdDevBright0.sum() / 5000) 
    VarOfStdDevBrightness    = StdDevBright0.var() 
    
    #TrainSet1
    MeanOfAverageBrightness1  = (NewArrayTrainSet1.sum() /5000)  
    VarOfAverageBrightess1    = NewArrayTrainSet1.var() 
    MeanOfStdDevBrightness1   = (stdDevArray1.sum() / 5000) 
    VarOfStdDevBrightness1    = stdDevArray1.var()

    
    
    #printTrainSetValues
    print(MeanOfAverageBrightness)
    print(VarOfAverageBrightess)
    
    print(MeanOfStdDevBrightness)
    print(VarOfStdDevBrightness)
    
    print(MeanOfAverageBrightness1)
    print(MeanOfStdDevBrightness1)
    
    print(VarOfAverageBrightess1)
    print(VarOfStdDevBrightness1)
   # I need to get the mean / standard deviation for each test1 image                       
   
    for i in range(980):
   #I need to store my Average Pixel Brightness/Std of test0                              - 
        NewArrayTestSet0[i]  = test0[i].mean(dtype = float) 
        stdDevArray2[i]      = test0[i].std(dtype = float) 
   
    
   #I need to get the mean/variation deviation of test0                                      
    NewArrayTestSet0Result     = NewArrayTestSet0.sum()/980
    NewArrayTestSet0ResultVar  = NewArrayTestSet0.var()
    stdDevArray2Result         = stdDevArray2.sum()/980
    stdDevArray2ResultVar      = stdDevArray2.var()
    
    #Print Values of TestSet0
    print('TestSet0')
    print(NewArrayTestSet0Result)
    print(NewArrayTestSet0ResultVar)
    print(stdDevArray2Result)
    print(stdDevArray2ResultVar)

   #I need to store my Average Pixel Brightness/Std of test1                             
    for x in range(1135):
   #  I need to store my Average Pixel Brightness of test2                              
        NewArrayTestSet1[x]  = test1[x].mean(dtype = float)
        stdDevArray3[x]      = test1[x].std(dtype  = float)
        
   #I need to get the mean/variation deviation of test1                                    
        
    MeanOfAverageBrightnessTest1  = NewArrayTestSet1.sum() /1135 
    VarOfAverageBrightessTest1    = NewArrayTestSet1.var()
    MeanOfStdDevBrightnessTest1   = stdDevArray3.sum() / 1135 
    VarOfStdDevBrightness1        = stdDevArray3.var()              
    
    print('TestSet1')
    print(MeanOfAverageBrightnessTest1)
    print(VarOfAverageBrightessTest1)
    print(MeanOfStdDevBrightnessTest1)
    print(VarOfStdDevBrightness1)
   
     
   
      
    
    #Calculate Probability Calculations Definition
    def CalculateProbability(sample:float, mean:float, variance:float):
        exponent  = numpy.zeros(1135)
        Prob      = numpy.zeros(1135)
        meanArray = numpy.ones_like(sample)*mean 
        FinalSize = numpy.zeros(1135)
        Diff      = numpy.zeros(1135)
        ExpCalc   = numpy.zeros(1135)
        
     # Calculate Probability Calculations
        Diff = numpy.power((sample - meanArray),2, dtype= float)
        ExpCalc = (-1*(Diff)/(2*variance))
        exponent = numpy.exp(ExpCalc, dtype = float)
        Prob = (exponent / (numpy.sqrt(2*numpy.pi * variance, dtype= float)))
        return Prob
    
    #Vectorize Probability Calculation
    vfunc = numpy.vectorize(CalculateProbability)
    
  
    
    # P(x|y) Probability Values for Each Y & Feature
    
    # TestProbabilities
# TestSet0 

    #TestSet0  for TrainSet0
    Feature1SuchY0ProbTrainSet0 = vfunc(NewArrayTestSet0,MeanOfAverageBrightness,VarOfAverageBrightess)  
    Feature2SuchY0ProbTrainSet0 = vfunc(stdDevArray2,MeanOfStdDevBrightness,VarOfStdDevBrightness)    
    #TestSet0  for TrainSet1
    Feature1SuchY0ProbTrainSet1 = vfunc(NewArrayTestSet0,MeanOfAverageBrightness1,VarOfAverageBrightess1)  
    Feature2SuchY0ProbTrainSet1 = vfunc(stdDevArray2,MeanOfStdDevBrightness1,VarOfStdDevBrightness1)         
    

    #DirectComparision of TstSet0|TrainSet0 vs #TstSet0|TrainSet1 = Choose Highest Probability for MeanBrightness(P(x)) 
    
 # TestSet1
    
    #TestSet1 for TrainSet0
    
    Feature1SuchY1ProbTrainSet0 = vfunc(NewArrayTestSet1,MeanOfAverageBrightness,VarOfAverageBrightess)
    Feature2SuchY1ProbTrainSet0 = vfunc(stdDevArray3,MeanOfStdDevBrightness,VarOfStdDevBrightness)
    
    #TestSet1 for TrainSet1
    Feature1SuchY1ProbTrainSet1 = vfunc(NewArrayTestSet1,MeanOfAverageBrightness1,VarOfAverageBrightess1)
    Feature2SuchY1ProbTrainSet1 = vfunc(stdDevArray3,MeanOfStdDevBrightness1,VarOfStdDevBrightness1)
 
    
    #Predictions
    
    # Expected to find 
    
    F1Y0=numpy.greater(Feature1SuchY0ProbTrainSet0,Feature1SuchY0ProbTrainSet1) 
    Num0Ttl = F1Y0.sum()
    Num0 = (Num0Ttl) 
    Num0diff = (980 - Num0) 
    
    F2Y0=numpy.greater(Feature2SuchY0ProbTrainSet0,Feature2SuchY0ProbTrainSet1) 
    Num1Ttl = F2Y0.sum()
    Num1 = (Num1Ttl) 
    Num1diff = (980 - Num0) 
    
    F1Y1=numpy.greater(Feature1SuchY1ProbTrainSet1,Feature1SuchY1ProbTrainSet0) 
    Num2Ttl = F1Y1.sum()
    Num2 = (Num2Ttl) 
    Num2diff = (980 - Num0) 
    
    F2Y1=numpy.greater(Feature2SuchY1ProbTrainSet1,Feature2SuchY1ProbTrainSet0) 
    Num3Ttl = F2Y1.sum()
    Num3 = (Num3Ttl) 
    Num3diff = (980 - Num0) 
    print('Choose0')
    print(Num0)
    print(Num1)
    print(Num2)
    print(Num3)

    print('AccuracyTest0')
    Num0/980
    
    print('AccuracyTest1')
    Num3/1135

if __name__ == '__main__':
    main()