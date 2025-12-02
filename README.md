**The BQ_2D code implement the Bayesian Quadrature (BQ), equipped with four new acuqisition functions, for estimating posteriors and the associated model evidence with desired accuracy. One can refer to Algorithm 1 in the following reference for details. Results in Figure 3-6 of this paper are produced with this code [2].   

**The TBQ_2D code implements the Transitional Bayesian Quadrature(TBQ) algorithm for Bayesian model inference, with the aim of estimating both posteriors and model evidences with desired accuracy. The method is capable of estimating these two terms with several dozens or at most several hundreds of likelihood calls, and epsecially effective for model with posteriors showing complex features like multy-modalities, high sharpness and nonlinear dependencies. The TBQ framework is originally developed in Ref.[1], and substentially improved in Ref.[2], with the developments of four new acquisition functions (new PUQ, PVC, PLUR and PEUR), and an alternative scheme for quantifying prediction uncertianties. This code implements the four two-dimensional exmaples in exmpale 1 of Ref. [2], where results in Figure 7-10 are produced with this code

Source articles
[1]. Wei, Pengfei. "Bayesian model inference with complex posteriors: Exponential-impact-informed Bayesian Quadrature." Mechanical Systems and Signal Processing 239 (2025): 113333.                                           
[2]. Song, Jingwen, and Wei, Pengfei. "Bayesian Model Inference using Bayesian Quadrature: the Art of Acquisition Functions and Beyond." arXiv preprint arXiv:2510.08974 (2025), submitted to Mechanical Systems and Signal Processing                                       


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

