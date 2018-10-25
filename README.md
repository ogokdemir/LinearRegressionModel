-----------------------

Say that we have two sections of an identical class. Students in one of the sections will take the test before the students in the second section. Assume further that before an exam is administered, for each student in the first section, data is collected on the number of hours each student studied for the test. After the first section takes the test we also collect information on who has passed the test and who has not. This data looks like the following,

	Hours 0.50 0.75 1.00 1.25 1.50 1.75 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 4.00 4.25 4.50 4.75 5.00 5.50
	Pass     0    0    0    0    0    0    1    0    1    0    1    0    1    0    1    1    1    1    1    1

(Note, this is the motivational example in the wikipedia article on  [Logistic Regression](http://en.wikipedia.org/wiki/Logistic_regression))

What we would like to do next is to make predictions on who will pass the test from the second section as we ask them how many hours each student has studied.

Notice that there are many ways to approach this problem.

## Examples of approaches that do not require a model ##

### Nearest neighbor. ###
For example, we can use the "nearest neighbor" approach and say that if a student in section 2 studied for 1.9 hours, then this student is most like the student in section 1 who studied for 2 hours. Thus, our prediction is that this student will not pass the test. However, if another student studied for 1.8 hours, we would say that this student is most like the two students who studied for 1.75 hours and thus is just as likely to pass as to fail.
This approach is not very commonly used in statistical learning. However, it does not necessitate a model and is pretty simple conceptually. Nonetheless, the results are overly flexible and do not include much structure from the data. This approach will easily overfit the data, producing predictions based on noise as much as the signal embedded in the data.

##Response: ##

-This is the most basic approach to the problem and is indeed easy to understand and implement. 
-Although this method seems to be reasonable in the realm of the outliers, it does not produce significantly accurate predictions near the median of the dataset.
-One of the main cons of this approach is that it does not involve a model. Therefore, it does not offer a gradual refinement of the prediction accuracy based on new data. 


### K nearest neighbors. ###
This approach requires us to look at k nearest neighbors in the data. For example if k=3, we would look at the three nearest neighbors and in our example of two students who studied 1.8 and 1.9 hours respectively, both would be given a one in three probability of passing the exam.
This approach also does not necessitate a model and is simple conceptually. However, it now starts to exhibit a little inference based on the structure of the data. However, the approach may also be too flexible in nature and thus open to ovefitting the data.

## Response: ## 

-This approach offers some improvemnt on the Nearest neighbour idea. However, it's success is heavily dependent on the structure of the data. 
-Like the nearest neighbor approach, this approach does not include a model, thus, we can't talk about a gradual refinement of the prediction accuracy. 
-This method is effective in the homogenous sections of the data, that is to say, when 1 and 0 values are congregated in a certain segment. It also improves on the 
first idea by proposing a probability in the domain of k-nearest neighbours rather than simply assigning the pass-fail value of the nearest neighbor. 
-This idea is not applicable to our dataset since, like the first model, this method will create clusters of same-value nodes in the dataset, and will lead to high probabilities 
of making the wrong prediction in the future. 

## Model based approaches ##
It is very common in statistical learning to assume that there is an underlying model that generated the data. For example, we can say that if a student studies for 1 hour or less their chances of passing the exam are one in 128. But, for every extra 45 minutes that a student studies, this student's chances of passing the exam doubles. Then we can go back and look at our data to see if our data supports this assumption.

### Objective function, part 1 ###
But, what does it mean that the data supports the model? 
Using our model, we now have a probability and we can use that probability to make a guess classification for the students in section 1. Then we can see how many times we were right and how many times we were wrong and decide if we are doing well in our prediction method based on these numbers.

## Response: ## 
This is indeed the contemporary approach in data science and is applicable to our problem. Presence of a model allows for training that model over an extended period of time with new data and improving on the prediction accuracy.
The only caveat with this particular objective function is that it contradicts with the essence of machine learning by assigning static and programmatical, instead of dynamic and probabilistic paramaters for the model. 



### Parameterized model. ###
Alternatively, we can avoid making an assumption that it was 45 minutes and assume that the number of minutes is nM. Then we can go back to our objective function and figure out which value for nM will provide us with the best result. This, nM, is a parameter of our model. Using a different value for nM changes our model.

### Objective function, part 2 ###
When a student passes the exam and we assigned a probability of .99 to this student, we may want to acknowledge that with a higher mark than if we had a lower probability. However, our objective function did not differentiate between a probability of 0.51 and the probability of 0.99 since both of these would produce the same classification of a pass. Consequently, we rework our objective function.

So, let's define a little notation. Let P(i) = 1 if i-th student passed the exam and P(i) = 0 if i-th student failed. While, pr(i,nM) is the probability that i-th student passed the exam, given a value for the parameter nM.

Now, for example, we can define the objective function as,

* Sum over all i of (P(i)-pr(i,nM))^2 

Minimizing this sum over different values of nM would provide us with a model.

### Logistic regression: (a type of a parameterized model) ###

	pr(i, beta0, beta1) = 1 / (1 + exp(-(beta0 + beta1*hours)))

This model guarantees that all probabilities are between 0 and 1.

##Response: ## 

This is indeed the most applicable and preferrable approach to our problem since it embodies dynamic parameters which reflect the data more intimately. 
Another main benefit of this objective function is that it provides a scale of the probability of the outcome instead of a binary pass or fail prediction, thus, proposing a substantial depth the to our analysis.
The model proposed above is simply a line-fitting application which aims to fit a line(whose equation is our model), to the data such that the line minimizes the cumulative regression of all data points in the set.


### Objective function, part 3 ###
Note, that there are many possible objective functions we can use. We will define two for this assignment (all using the logistic function above):

* A) Minimize Sum (over all i) of (P(i)-pr(i,parameters))^2

	[LaTeX: \sum_i (P(i)-pr(i,\beta_0,\beta_1))^2 ]

##Response: ## 
This approach is plausible since it embodies the idea of linear regression line. In this approach we are basically trying to find parameters (beta0,beta1) such that we minimize the 
inaccuracy of the predictions of our model. It is easy to implement and we believe that this approach is plausible for implementation in order to solve the problem at hand.


* B) Maximize (Product (over all i that passed) of pr(i,parameters))*(Product (over all i that did not pass) of (1-pr(i,parameters)))  

	[LaTeX: \prod_{i\in Passed}pr(i,\beta_0,\beta_1) \prod_{i\in Failed}(1-pr(i,\beta_0,\beta_1))  ]

* B.ii) This is an equivalent formulation to (B), but you may find it easier to perform the maximization on the log, e.g., Maximize (Sum (over all i that passed) of log(pr(i,parameters)))+(Sum (over all i that did not pass) of log(1-pr(i,parameters)))  

	[LaTeX: \sum_{i\in Passed}ln(pr(i,\beta_0,\beta_1)) \sum_{i\in Failed}ln(1-pr(i,\beta_0,\beta_1)) ]

## Response: ##

This approach is commensurate with the previous objective function. The main difference is: the previous function was seeking parameters (beta0,beta1) such that the linear regression line 
(whose equation is our prediction model) minimizes the regression, i.e, overall inaccuracy of the predictions whereas this function aims to maximize the overall accuracy of the predictions
of our model. It is easy to program and understand, thus, is a strong candidate for our implementation.


