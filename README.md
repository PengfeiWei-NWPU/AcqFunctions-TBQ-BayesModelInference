Run the BQ_2D code, it is required to input the index of problem to solved and the index of acquisition function to be used. For example, with the following inputs,
Input the index of problem to be solved: 2
Input the index of Acqusition function (0. PUQ (old version) 1. PUQ (new version), 2. PVC, 3. PLUR, 4. PEUR):2
The following results are produced for estimating model evidence,
Total number of likelihood calls： 31
Mean estimate of model evidence： 0.1282
COV of the mean estimate： 0.0000
Reference value of model evidence computed by MC： 0.1254
CoV of reference estimate by MC： 0.0022
, together with a figure for showing results of the posterior as:
![BQresults](https://github.com/user-attachments/assets/7f60ee4e-c277-457a-8b61-2d834e2f1f89)


Run the TBQ_2D code, and input the index of problem and the index of acquisition function as follows:
Input the index of problem to be solved: 4
Input the index of Acqusition function (0. PUQ (old version), 1. PUQ (new version), 2. PVC, 3. PLUR, 4. PEUR):4
the following results are produced:
Mean estimate of model evidence（No bridging）： 0.22237981
Mean estimate of model evidence（with bridging）： 0.22655018
Reference value of model evidence： 0.22884300
Post CoV of estimate： 0.0052
Number of model calls： 75
gamma value:0.0000  0.0432  0.1230  0.3355  0.4971  1.0000  
Number of accumulated function calls consumed by each tempering stage:12.0000  27.0000  34.0000  45.0000  63.0000  75.0000  
 Estimation of ratios between consecutive tempering stages (no bridging):1.0000  0.6109  0.7034  0.7206  0.8458  0.8492  
 Estimation of ratios between consecutive tempering stages (with bridging):1.0000  0.6150  0.7080  0.7207  0.8460  0.8532  
 , accompanied with the contour plots of intermediate posteriors, as well as the MCMC samples of each intermediate stage, as follows
 
![TBQresults_contourplots](https://github.com/user-attachments/assets/d6f10e96-68da-4528-8c88-489d39b46e52)

![TBQresults_MCMCsamples](https://github.com/user-attachments/assets/851da624-faaa-408b-9a3f-543da6f8fe1c)

