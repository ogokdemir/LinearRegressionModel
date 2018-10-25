import numpy as np
import math 
import sys

#reading the file.
file = open("data.txt", "r")
next(file)

#cache that maps the hours of study to the pass-fail results.
sectionOneResults = {}

#iterate through the file and inflate the cache. 
for row in file:
   temp = row.split(",")
   hours = float(temp[0])
   passed = float(temp[1])
   sectionOneResults[hours] = passed





'''
This is the logictic regression function we are using to calculate probabilities of passing.
This function is guaranteed to produce probabilities that range between 0 and 1.

#params:  hours, numbet of hours studied 
          beta0, beta1 are the parameters of the model we aim to optimize. In other words, these are the coefficient in the equation of linear regression line. 

#returns: probability of a student passing the test given the model params(beta0 and beta1) and number of hours they studied.
'''

def logisticRegression(hours, beta0, beta1):
    return 1 / (1 + math.exp(-(beta0 + beta1*hours)))




'''
Choice of Objective Function and Reasoning:

We chose to utilize the approach of minimizing the sum of P(i)-pr(i,params) values for all i, i.e, all datapoints. We aver that this is a plausible approach because:

Assuming that P(i)=0 for a student i who failed, and P(i)=1 for a student that passed;


a) if P(i)=0, then P(i)-pr(i,params) value corresponds to the inaccuracy of our prediction because pr is the predicted probability of the student's passing.
                                     The bigger the pr value, the more significant of an error our model made. Furthermore, smaller the predicted
                                     probability of passing, P(i)-pr(i,params) will be smaller, which could be interpreted as we made a correct prediction
                                     of failure by assigning a low probability of passing. 

b) elif P(i)=1, then P(i)-pr(i,params) value corresponds to the strength of accuracy. The smaller this value, the better of a prediction our model made. 
                                       because the higher the predicted probability of passing, P(i)-pr(i,params) will be smaller, i.e, We were confident in our prediction that
                                       the student would pass and we indeed made the right prediction.
                                       
                                       also, smaller the predicted probability of passing, P(i)-pr(i,params) will be larger, which could be interpreted as we made a wrong
                                       prediction by calculating a smaller probability of passing when the student actually passed.



Conclusion: In the light of the understandings discussed above, we are confident that this objective function will minimize our error rate and therefore will facilitate a
line-fitting process which will culminate optimum coefficients for our model based on numerous random options. We will simply generate random values for beta0 and beta1
then we will test these random values and pick the pair of coefficients for our model which minimizes the error(regression.)

'''


'''
@params: data, this is the entire dataset in the form of a dictionary that maps the hours of studying to a binary pass-fail outcome.
@returns: optimum pair of (beta0,beta1) coefficients in a tuple.
'''

#This dictionary stores the error rate for each random pair of coefficients in the linear regression line.
#We will retain the pair that produced the minimum error rate. This pair is the optimum coeffiencts in the equation of our  linear regression line.



sumErrorToParams = {}

def objectiveFunction(data):

    for hours,passed in data.items():
        #try 1000 beta0 values ranging between -10 and 10
        for beta0 in np.random.uniform(-10,10,1000):
            #try 1000 beta1 values ranging between -10 and 10
            for beta1 in np.random.uniform(-10,10,1000):
                sumErrorToParams[(beta0, beta1)] = abs(passed - logisticRegression(hours, beta0, beta1))



#Client 
def main():
    
    objectiveFunction(sectionOneResults)
    
    currentBest = ((0,0), sys.maxsize)

    for k,v in sumErrorToParams.items():
        if v < currentBest[1]:
            currentBest = (k,v)
            
    print(currentBest)


'''
A comment on the results:

We observed that when the broaden the range of possible values for beta0 and beta1, we get better models(with less regression).
We also observed that when we increase the number of trials, despite the increase in complexity by O(n^2), we refine better models. 
'''

main()



