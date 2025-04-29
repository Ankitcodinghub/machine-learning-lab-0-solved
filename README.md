# machine-learning-lab-0-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Lab 0 Solved](https://www.ankitcodinghub.com/product/machine-learning-labs-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110039&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning  Lab 0 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Introduction

For computational efficiency of typical operations in machine learning applications, it is very beneficial to use NumPy arrays together with vectorized commands, instead of explicit for loops. The vectorized commands are better optimized, and bring the performance of Python code (and similarly e.g. for Matlab) closer to lower level languages like C. In this exercise, you are asked to write efficient implementations for three small problems that are typical for the field of machine learning.

Getting Started

Follow the Python setup tutorial provided on our github repository here:

github.com/epfml/ML course/tree/master/labs/ex01/python setup tutorial.md

After you are set up, clone (using command line or a git desktop client) or download the repository, and start by filling in the template notebooks in the folder /labs/ex01, for each of the 3 tasks below.

To get more familiar with vector and matrix operations using NumPy arrays, it is also recommended to go through the npprimer.ipynb notebook in the same folder.

Note: The following three exercises could be solved by for-loops. While that’s ok to get started, the goal of this exercise sheet is to use the more efficient vectorized commands instead:

Useful Commands

We give a short overview over some commands that prove useful for writing vectorized code. You can read the full documentation and examples by issuing help(func).

At the beginning: import numpy as np

• a * b, a / b: element-wise multiplication and division of matrices (arrays) a and b

• a.dot(b): matrix-multiplication of two matrices a and b

• a.max(0): find the maximum element for each column of matrix a (note that NumPy uses zero-based indices, while Matlab uses one-based)

• a.max(1): find the maximum element for each row of matrix a

• np.mean(a), np.std(a): compute the mean and standard deviation of all entries of a

• a.shape: return the array dimensions of a

• a.shape[k]: return the size of array a along dimension k

• np.sum(a, axis=k): sum the elements of matrix a along dimension k

• linalg.inv(a): returns the inverse of a square matrix a

A broader tutorial can be found here: https://sites.engineering.ucsb.edu/~shell/che210d/numpy.pdf

For users who were more familiar with Matlab, a nice comparison of the analogous functions can be found here:

https://numpy.org/devdocs/user/numpy-for-matlab-users.html

Figure 1: Two sets of points in the plane. The circles are a subset of the dots and have been perturbed randomly.

Task A: Matrix Standardization

The different dimensions or features of a data sample often show different variances. For some subsequent operations, it is a beneficial preprocessing step to standardize the data, i.e. subtract the mean and divide by the standard deviation for each dimension. After this processing, each dimension has zero mean and unit variance. Note that this is not equivalent to data whitening, which additionally de-correlates the dimensions (by means of a coordinate rotation).

Write a function that accepts data matrix x ∈ Rn×d as input and outputs the same data after normalization. n is the number of samples, and d the number of dimensions, i.e. rows contain samples and columns features.

Task B: Pairwise Distances in the Plane

One application of machine learning to computer vision is interest point tracking. The location of corners in an image is tracked along subsequent frames of a video signal (see Figure 1 for a synthetic example). In this context, one is often interested in the pairwise distance of all points in the first frame to all points in the second frame. Matching points according to minimal distance is a simple heuristic that works well if many interest points are found in both frames and perturbations are small.

Write a function that accepts two matrices P ∈ Rp×2,Q ∈ Rq×2 as input, where each row contains the (x,y) coordinates of an interest point. Note that the number of points (p and q) do not have to be equal. As output, compute the pairwise distances of all points in P to all points in Q and collect them in matrix D. Element Di,j is the Euclidean distance of the i-th point in P to the j-th point in Q.

Task C: Likelihood of a Data Sample

In this exercise, you are not required to understand the statistics and machine learning concepts described here yet. The goal here is just to practically implement the assignment of data to two given distributions, in Python.

A subtask of many machine learning algorithms is to compute the likelihood p(xn|θ) of a sample xn for a given density model with parameters θ. Given k models, we now want to assign xn to the model for which the likelihood is maximal: an = argmaxm p(xn |θm), where m = 1,…,k. Here θm = (µm,Σm) are the parameters of the m-th density model (µm ∈ Rd is the mean, and Σm is the so called covariance matrix).

We ask you to implement the assignment step for the two model case, i.e. k = 2. As input, your function receives a set of data examples xn ∈ Rd (indexed by 1 ≤ n ≤ N) as well as the two sets of parameters θ1 = (µ1,Σ1) and θ2 = (µ2,Σ2) of two given multivariate Gaussian distributions:

.

2

Theory Questions

In addition to the practical exercises you do in the labs, as for example above, we will in future labs also provide you some theory oriented questions, to prepare you for the final exam. As the rest of the exercises, it is not mandatory to solve them, but we would recommend that you at least look at – and try – some of them during the semester. From experience, students are sometimes surprised by the heavy theoretical focus of the final exam after having worked on the two very practical projects. Do not fall for this trap! Passing the course require acquiring both a practical and a theoretical understanding of the material, and theory exercises should help you with the latter.

If you are looking for additional resources for additional theory questions to practice, there are several (optional) excellent books and online resources – see for example in the The course info sheet.

This week, as we just started the course, there are no exercises. You should instead take some time to refresh your mind about the prerequisites, especially on the following topics.

• Make sure your linear algebra is fresh in memory, especially

– Matrix manipulation (Multiplication, Transpose, Inverse)

– Ranks, Linear independence

– Eigenvalues and Eigenvectors

You can use the following resources to help you get up to speed if needed.

– The Linear Algebra handout

– Gilbert Strang’s Linear Algebra and Learning from Data or Introduction to Linear Algebra. Some chapters are available online, and the books (along with many other textbooks on linear algebra) should be available at the EPFL Library.

• If it has been long since your last calculus class, make sure you know how to handle Gradients. You can find a quick summary and useful identities in The Matrix Cookbook.

• For probability and statistics, you should at least know about

– Conditional and joint probability distributions

– Bayes theorem

– Random variables, independence, variance, expectation

– The Gaussian distribution

If you need a refresh, check Chapter 2 in Pattern Recognition and Machine Learning by Christopher Bishop, available at the EPFL Library.

3
