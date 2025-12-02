%% this code implements the Transitional Bayesian Quadrature(TBQ) algorithm for Bayesian model inference, 
%% with the aim of estimating both posteriors and model evidences with desired accuracy. The method is 
%% capable of estimating these two terms with several dozens or at most several hundreds of likelihood calls, 
%% and epsecially effective for model with posteriors showing complex features like multy-modalities, high sharpness 
%% and nonlinear dependencies. 
%% The TBQ framework is originally developed in Ref.[1], and substentially improved in Ref.[2], with the developments of four 
%% new acquisition functions (new PUQ, PVC, PLUR and PEUR), and an alternative scheme for quantifying prediction uncertianties 
%% This code implements the four two-dimensional exmaples in exmpale 1 of Ref. [2], where results in Figure 7-10 are produced with this code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Source articles%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[1]. Wei, Pengfei. "Bayesian model inference with complex posteriors: Exponential-impact-informed Bayesian Quadrature."                  %%%%%%
%     Mechanical Systems and Signal Processing 239 (2025): 113333.                                                                        %%%%%%
%[2]. Song, Jingwen, and Wei, Pengfei. "Bayesian Model Inference using Bayesian Quadrature: the Art of Acquisition Functions and Beyond."  %%%%%%
%     arXiv preprint arXiv:2510.08974 (2025), submitted to Mechanical Systems and Signal Processing                                       %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
%%Source of examples:Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." International conference on machine learning. PMLR, 2015. 
%%Define the four energy functions
w1 = @(x)sin(2*pi*x(:,1)/4);
w2 = @(x)3*exp(-0.5*((x(:,1)-1)/0.6).^2);
w3 = @(x)3./(1+exp(-(x(:,1)-1)/0.2));
U{1} = @(x) 0.5*((sqrt(sum(x(:,1:2).^2,2))-2)/0.4).^2 - log(exp(-0.5*(x(:,1)-2).^2/0.6^2)+exp(-0.5*(x(:,1)+2).^2/0.6^2));
U{2} = @(x) 0.5*((x(:,2)+w1(x))/0.4).^2;
U{3} = @(x) -log(exp(-0.5*((x(:,2)+w1(x))/0.35).^2)+exp(-0.5*((x(:,2)+w1(x)-w2(x))/0.35).^2));
U{4} = @(x) -log(exp(-0.5*((x(:,2)+w1(x))/0.4).^2)+exp(-0.5*((x(:,2)+w1(x)-w3(x))/0.35).^2));


IndProb = input('Input the index of problem to be solved: ');%%% there are four example, input the index of problem to be solved
AcqFunIndex = input('Input the index of Acqusition function (0. PUQ (old version), 1. PUQ (new version), 2. PVC, 3. PLUR, 4. PEUR):');%% input the index of acquisition function to be used

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LogLikelihood =@(z)-U{IndProb}(z);%define the log of likelihood
Likelihood = @(z)exp(LogLikelihood(z));% define the likelihood
nz = 2;%dimension of parameters
lb = -4; ub = 4; %lower and upper bounds of prior distribution
PriorPDF = @(z)unifpdf(z(:,1),lb,ub).*unifpdf(z(:,2),lb,ub);% Define the prior

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Specify the algorithm parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N0 = 12; %%Number of initial sample size
NMC = 5e3; %sample size  
zeta = 0.2;% scale parameter of the proposal distribution for MH sampling 

if IndProb == 1|| IndProb ==2%%%parameter settings for diferent examples
   KerFun = 'ardsquaredexponential';%%kernel function for training GP, available: 'ardsquaredexponential',  'ardexponential','ardmatern32','ardmatern52'
   CV = 1; %parameters for controlling the divergence of consecutive intermediate posteriors, recomendation 0.5~1
   StopThreshold_Int = 0.04;% Stopping threshold for Z_j/Z_j-1 (intermediate stages),
   StopThreshold_Final = 0.04;% Stopping threshold for Z_j/Z_j-1 (final stage) 
   FixedThreshold = 0.2;% in case he posterior covariance touch this value for two times consecutively, 
                     % fix the value of gamma_Active to improve the exploitation performance, 
                     % this can be generally positive for improve convergence speed, but 
   Lchain = 20; % length of each Markov Chain, sugegsted to be 20~50, to avoid particle degradation and skipping the burn-in period
   StopLimit = 2;%% delayed judgement
elseif IndProb == 3
   KerFun =  'ardsquaredexponential';%%kernel function for training GP, available: 'ardsquaredexponential',  'ardexponential','ardmatern32','ardmatern52'
   CV = 1; %parameters for controlling the divergence of consecutive intermediate posteriors
   StopThreshold_Int = 0.02;% Stopping threshold for Z_j/Z_j-1 (intermediate stages)
   StopThreshold_Final = 0.02;% Stopping threshold for Z_j/Z_j-1 (final stage)
   FixedThreshold = 0.1;% in case he posterior covariance touch this value for two times consecutively, 
                     % fix the value of gamma_Active to improve the exploitation performance, 
                     % this can be generally positive for improve convergence speed, but 
   Lchain = 20; % length of each Markov Chain, sugegsted to be 20~50, to avoid particle degradation and skipping the burn-in period
   StopLimit = 2;%% delayed judgement
else
    KerFun = 'ardmatern52';%%kernel function for training GP, available: 'ardsquaredexponential',  'ardexponential','ardmatern32','ardmatern52'
    CV = 0.5; %parameters for controlling the divergence of consecutive intermediate posteriors
    StopThreshold_Int = 0.02;% Stopping threshold for Z_j/Z_j-1 (intermediate stages)
    StopThreshold_Final = 0.02;% Stopping threshold for Z_j/Z_j-1 (final stage)
    FixedThreshold = 0.02;% in case he posterior covariance touch this value for two times consecutively, 
                     % fix the value of gamma_Active to improve the exploitation performance, 
                     % this can be generally positive for improving
                     % convergence speed, but be careful with the
                     % specification of this value
    Lchain = 20; % length of each Markov Chain, sugegsted to be 20~50, to avoid particle degradation and skipping the burn-in period
    StopLimit = 2;%% delayed judgement
end
%for example 1 and 2, 'ardsquaredexponential' can be used, but for the
%later two, matern32 is suggested
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%compute the referenc value of modle evidence by MC, only for comparison, if not required, skip this part to save time%%%%%%%%%%
Ntest = 1e6;
for i=1:nz
   TestSamp(:,i) = unifrnd(lb,ub,Ntest,1);
end
ZRef = mean(Likelihood(TestSamp)); 
%%%%%%Initialize the algorithm

MCSamp(:,:,1) = unifinv(lhsdesign(NMC,nz),lb,ub);%the first sample populaion for the first tempering stage
RandInd = randperm(NMC,NMC);% Indexes of randomly permuting the row of MCSamp to generate the second sample population
MCSamp2(:,:,1) = MCSamp(RandInd,:,1);%%%the second sample population for the first tempering stage
Ttrain0 = unifinv(lhsdesign(N0,nz),lb,ub);%initial training samples of theta


Ltrain0 = LogLikelihood(Ttrain0); % compute the values of the logarithm of the likelihood 
Ttrain = Ttrain0;%initialize the training data
Ltrain = Ltrain0;
Ncall = N0;%used for record the number of accumulated training samples 
NcallEachStage(1) = N0;% record the number of training points for each stage
j=1;%index of tempering stage
gamma_fixed(j) = 0;%initialzie the value of tempering parameter
LikeFun_Fixed{j} = @(z)ones(size(z,1),1);%%initialize the fixed intermediate likelihood scaled by gamma_j
LikeSamp(:,j) = ones(NMC,1);%%initial the values of intermediate likelihood indexed by gamma_j
Zratio(j) = 1;%%initialize the ratio of intermediate normalizing constants, estimated by MC scheme
Zratio_bridg(j) = 1;%%initialize teh ratio of intermediate normalizing constants Z_j/Z_j-1, estimated by a bridging MC scheme, see Ref. [2]

while gamma_fixed(j)<1 %% in case gamma_j is smaller than 1
    j=j+1;
    StopFlag = 0;
    FixedFlag = 0;
    TrainInd = [];
    while 1==1%% for the j-th stage
        GPRmodel = fitrgp(Ttrain,Ltrain,'KernelFunction',KerFun...
                   ,'BasisFunction','constant','Sigma', 1e-4, 'ConstantSigma', true,'SigmaLowerBound', 1e-5);  
        [MeanPred,SDPRed] = predict(GPRmodel,MCSamp(:,:,j-1));
        SampPred = [MeanPred,SDPRed];
        %%%%%%%%%%%%%Updating gamma_Active%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        weight = @(gamma)exp(gamma*MeanPred)./LikeSamp(:,j-1);%%define the ratio of consecutive likelihoods for MCSamp(:,:,j-1)
        weightLim = weight(1);%% weights of samples computed at gamma=1
        if FixedFlag<2% in case gamma_Active should not be fixed, update it within each iteration, this is benificial for improving the exploration performance
           if std(weightLim)/mean(weightLim)>CV% in case gamma_Active is smaller than one and needs to be updated
              gamma_Active = fminbnd(@(gamma)abs(std(weight(gamma))/mean(weight(gamma))-CV),gamma_fixed(j-1),1);
           else% in case the maximum COV is lower than CV, set gamma as 1
              gamma_Active = 1;
           end
        end
        fprintf('Active gamma value： %.4f\n', gamma_Active);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if gamma_Active==1% specify different stopping thresholds for intermediate and final stages
            StopThreshold = StopThreshold_Final;
        else 
            StopThreshold = StopThreshold_Int;
        end
       
        % DeltaGamma = gamma_Active - gamma_fixed(j-1);
        %%%%%%%%%%%%%compute the mean and variance of Z_j/Z_j-1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       [~,covmat,~] = predictExactWithCov(GPRmodel.Impl,MCSamp(:,:,j-1),0.05);
       indices = sub2ind(size(covmat), 1:1:NMC, RandInd);
       CovValue = covmat(indices)';
       weight_Active = exp(gamma_Active*MeanPred + gamma_Active.^2.*SDPRed.^2/2)./LikeSamp(:,j-1);%Mean estimate of the ratio between two consecutive likelihoods
       MeanZratio = mean(exp(gamma_Active*MeanPred)./LikeSamp(:,j-1));%mean estimate of Z_j/Z_j-1
       VarZratio = mean(weight_Active.*weight_Active(RandInd,:).*(exp(gamma_Active.^2*CovValue)-1));%%Variance of Z_j/Z_j-1
       CoVZratio = sqrt(VarZratio)/MeanZratio;%coefficient of variation
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       fprintf('COV： %.4f\n', CoVZratio);
       fprintf('Ncall： %d\n', Ncall);
       %%%%%%%%%%%%%%next judge if the stopping condition for the j-th
       %%%%%%%%%%%%%%stage is satisfied for "StopLimit" times, if yes, skip out

       if CoVZratio<FixedThreshold%%judge whether to fix gamma_j
           FixedFlag = FixedFlag+1;
       else
           if FixedFlag==1
              FixedFlag=0;
           end
       end

       if CoVZratio<StopThreshold
           StopFlag = StopFlag+1;
       else
           StopFlag = 0;
       end
       if StopFlag==StopLimit
           break;
       else%%%if stopping condition not satisfied, query the next training point for updating, using one of the acquisition function
           if AcqFunIndex == 0
               AcqSamp = AcqFunTypeI(GPRmodel,gamma_Active,MCSamp(:,:,j-1),PriorPDF);
           elseif AcqFunIndex == 1
               AcqSamp = PUQFun(GPRmodel,gamma_Active,MCSamp(:,:,j-1),PriorPDF,LikeSamp(:,j-1));
           elseif AcqFunIndex == 2
               AcqSamp = PVCFun(GPRmodel,gamma_Active,MCSamp(:,:,j-1),PriorPDF,LikeSamp(:,j-1));
           elseif AcqFunIndex == 3
               AcqSamp = PLURFun(GPRmodel,gamma_Active,MCSamp(:,:,j-1),PriorPDF,LikeSamp(:,j-1));
           else
              AcqSamp = PEURFun(GPRmodel,gamma_Active,MCSamp(:,:,j-1),RandInd,LikeSamp(:,j-1));
           end
           AcqSamp(TrainInd) = -Inf;% avoid replication
           [~,Ind] = max(AcqSamp);
           TrainInd = [TrainInd,Ind];
           Tnew = MCSamp(Ind,:,j-1);
           Lnew = LogLikelihood(Tnew);
           Ttrain = [Ttrain;Tnew];
           Ltrain = [Ltrain;Lnew];
           Ncall = Ncall+1;
       end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    %%%after the convergence for the j-th stage, record the fixed quantities 
    NcallEachStage(j) = Ncall;%record the number of accumulated training points for the j-th stage
    GPmodel_fixed{j} = GPRmodel;
    gamma_fixed(j) = gamma_Active;
    LikeFun_Fixed{j} = @(z)LikeFun(z,GPmodel_fixed{j},gamma_fixed(j));

    Zratio(j) = mean(LikeFun_Fixed{j}(MCSamp(:,:,j-1))./LikeSamp(:,j-1));

    %%%%%%%%Next perform R-MH to sample from LikeFun_Fixed{j}*PriorPDF
    weight_j = LikeFun_Fixed{j}(MCSamp(:,:,j-1))./LikeSamp(:,j-1);%Mean estimate of the ratio between two consecutive likelihoods
    NormWeight = weight_j/sum(weight_j);% Compute the normalized weights for each sample in MCSamp(:,:,j-1)
    SampInd = randsample(1:1:NMC,NMC,true,NormWeight);% resampling with replication
    InitialSamp = MCSamp(SampInd,:,j-1);%generate initial training samples following h_j
    WeightedMean = sum(repmat(NormWeight,1,nz).* MCSamp(:,:,j-1),1);%weighted mean
    Deviation = sqrt(repmat(NormWeight,1,nz)).*(MCSamp(:,:,j-1) - repmat(WeightedMean,NMC,1));%weighted deviation of samples
    WeightedCOV = Deviation'*Deviation;%compute the covariance of samples with weights
    PropSigma = zeta^2*WeightedCOV;
    clear UnNormWeight SumWeight Weight SampInd_Record SampInd
    for k=1:NMC
          Chain{k}(1,:) = InitialSamp(k,:);%%seed of the k-th chain
    end
    parfor k=1:NMC
          for s = 2:Lchain% grow the s-th chain with seed Chain{k}(1,:), where Lchain can be set as 3~5, the higher the less replications of samples
              Xcand = mvnrnd(Chain{k}(s-1,:),PropSigma);%generate a candidate sample 
              alpha = LikeFun_Fixed{j}(Xcand).*PriorPDF(Xcand)./(LikeFun_Fixed{j}(Chain{k}(s-1,:)).*PriorPDF(Chain{k}(s-1,:))); % compute the acceptance probability of Xcand
              if min(alpha,1)>rand
                  Chain{k}(s,:) = Xcand;
              else
                  Chain{k}(s,:) = Chain{k}(s-1,:);
              end
          end
   end%end parfor for growing chains
   for k=1:NMC%last point of each chain as a sample
       MCSamp(k,:,j) = Chain{k}(Lchain,:);
   end
   clear Chain InitialSamp
   LikeSamp(:,j) = LikeFun_Fixed{j}(MCSamp(:,:,j));%record the  
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%Next re-estimate Z_j/Z_j-1 with bridging estimators%%%%%%%%%%%%%%%%
   Like_bridg_jminus1 = sqrt(LikeFun_Fixed{j}(MCSamp(:,:,j-1))./LikeSamp(:,j-1));
   Like_bridg_j = sqrt(LikeFun_Fixed{j-1}(MCSamp(:,:,j))./LikeSamp(:,j));
   Zratio_bridg(j) = mean(Like_bridg_jminus1)/mean(Like_bridg_j);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
Post_STDPred = @(z)LikeFunSTD(z,GPRmodel).*PriorPDF(z);%the posterior STD of the unnormalized posterior

%%%store the MCMC samples of each tempering stage, as well as the training data
if IndProb==1 
    if AcqFunIndex == 0
       writematrix(MCSamp,'TMC_U1_PUQOld.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PUQOld.xlsx')
    elseif AcqFunIndex == 1
       writematrix(MCSamp,'TMC_U1_PUQ.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PUQ.xlsx')
    elseif AcqFunIndex == 2
       writematrix(MCSamp,'TMC_U1_PVC.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PVC.xlsx')
    elseif AcqFunIndex == 3
       writematrix(MCSamp,'TMC_U1_PLUR.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PLUR.xlsx')
    else
       writematrix(MCSamp,'TMC_U1_PEUR.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PEUR.xlsx')
    end
elseif IndProb==2
     if AcqFunIndex == 0
       writematrix(MCSamp,'TMC_U1_PUQOld.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PUQOld.xlsx')
    elseif AcqFunIndex == 1
       writematrix(MCSamp,'TMC_U2_PUQ.xlsx')
       writematrix(Ttrain,'Ttrain_U2_PUQ.xlsx')
    elseif AcqFunIndex == 2
       writematrix(MCSamp,'TMC_U2_PVC.xlsx')
       writematrix(Ttrain,'Ttrain_U2_PVC.xlsx')
    elseif AcqFunIndex == 3
       writematrix(MCSamp,'TMC_U2_PLUR.xlsx')
       writematrix(Ttrain,'Ttrain_U2_PLUR.xlsx')
    else
       writematrix(MCSamp,'TMC_U2_PEUR.xlsx')
       writematrix(Ttrain,'Ttrain_U2_PEUR.xlsx')
    end
elseif IndProb==3
     if AcqFunIndex == 0
       writematrix(MCSamp,'TMC_U1_PUQOld.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PUQOld.xlsx')
    elseif AcqFunIndex == 1
       writematrix(MCSamp,'TMC_U3_PUQ.xlsx')
       writematrix(Ttrain,'Ttrain_U3_PUQ.xlsx')
    elseif AcqFunIndex == 2
       writematrix(MCSamp,'TMC_U3_PVC.xlsx')
       writematrix(Ttrain,'Ttrain_U3_PVC.xlsx')
    elseif AcqFunIndex == 3
       writematrix(MCSamp,'TMC_U3_PLUR.xlsx')
       writematrix(Ttrain,'Ttrain_U3_PLUR.xlsx')
    else
       writematrix(MCSamp,'TMC_U3_PEUR.xlsx')
       writematrix(Ttrain,'Ttrain_U3_PEUR.xlsx')
    end
else
     if AcqFunIndex == 0
       writematrix(MCSamp,'TMC_U1_PUQOld.xlsx')
       writematrix(Ttrain,'Ttrain_U1_PUQOld.xlsx')
    elseif AcqFunIndex == 1
       writematrix(MCSamp,'TMC_U4_PUQ.xlsx')
       writematrix(Ttrain,'Ttrain_U4_PUQ.xlsx')
    elseif AcqFunIndex == 2
       writematrix(MCSamp,'TMC_U4_PVC.xlsx')
       writematrix(Ttrain,'Ttrain_U4_PVC.xlsx')
    elseif AcqFunIndex == 3
       writematrix(MCSamp,'TMC_U4_PLUR.xlsx')
       writematrix(Ttrain,'Ttrain_U4_PLUR.xlsx')
    else
       writematrix(MCSamp,'TMC_U4_PEUR.xlsx')
       writematrix(Ttrain,'Ttrain_U4_PEUR.xlsx')
    end
end


fprintf('Mean estimate of model evidence（No bridging）： %.8f\n', prod(Zratio));
fprintf('Mean estimate of model evidence（with bridging）： %.8f\n', prod(Zratio_bridg));
fprintf('Reference value of model evidence： %.8f\n', ZRef);
fprintf('Post CoV of estimate： %.4f\n',  CoVZratio);
fprintf('Number of model calls： %d\n',  Ncall);
fprintf('gamma value:');
fprintf('%.4f  ',  gamma_fixed);
fprintf('\nNumber of accumulated function calls consumed by each tempering stage:');
fprintf('%.4f  ',  NcallEachStage);
fprintf('\n Estimation of ratios between consecutive tempering stages (no bridging):');
fprintf('%.4f  ',  Zratio);
fprintf('\n Estimation of ratios between consecutive tempering stages (with bridging):');
fprintf('%.4f  ',  Zratio_bridg);
fprintf('\n:');  

%%%%%%%%%%%%%%%%%%%%define colors for plotting%%%%%
C10 =[056 076 159; 094 147 195;157 206 226; 222 242 182;254 249 182;253 204 125;247 135 078;220 059 044;165 000 038]/255; %%适合pcolor
C11 = [012 007 134;076 002 161; 126 003 167;169 035 149;203 071 119;229 108 091; 248 149 064; 253 196 039;239 248 033]/255;%%适合pcolor
C12 = [002 048 071;014 091 118;026 134 163; 070 172 202; 155 207 232;243 249 252;255 202 095;254 168 009;253 152 002;251 132 002]/255;
C13 = [078 098 171; 070 158 180; 135 207 164; 203 233 157;245 251 177;254 232 154;253 185 106;245 117 071;214 064 078; 158 001 066 ]/255;
C14 = [026 049 139;073 108 206;130 170 231;185 210 243; 230 240 254; 249 219 229;247 166 191;228 107 144;192 063 103;154 019 061]/255;
C15 = [000 070 041;012 113 059;055 158 084;119 197 120;186 226 148;236 247 177;254 254 227]/255;
C16 = [090 095 163;143 152 201;191 192 222;224 210 229;231 203 224]/255;
C17 = [255 182 30;255 199 115;136 173 166;72 192 163;33 166 117;22 133 169;0 52 114]/255;%%千里江山图
blue1 = [0, 47, 167]/255; blue2 = [0, 49, 83]/255; blue3 = [129, 216, 208]/255;
yellow1 = [249, 220, 36]/255;yellow2 = [232, 88, 39]/255;
red1 = [231, 31, 24]/255; red2 = [76,0,9]/255;red3 = [128, 0,32]/255; red4  =[176,89,35]/255;
brown1 = [143, 75, 40]/255;
green1 = [1, 132,126]/255;
%%%%%%%%%Next plot the results%%%%%%%%
%%%%intermedaite posterior densities (unnormalzied)
%%%%STD of the final
%%%%training points for each stage, (make a difference between initial training points and actively designed points)
%%%%reference posterior
Nplot = 200;
Xplot = linspace(-4,4,Nplot);
[Xgrid,Ygrid] = meshgrid(Xplot,Xplot);
for i=1:Nplot
  PostRefGrid(:,i) = Likelihood([Xgrid(:,i),Ygrid(:,i)]).*PriorPDF([Xgrid(:,i),Ygrid(:,i)]);%%reference of unnormalzied posterior
  for s = 2:j
      PostMeanPred(:,i,s) = LikeFun_Fixed{s}([Xgrid(:,i),Ygrid(:,i)]).*PriorPDF([Xgrid(:,i),Ygrid(:,i)]);%%Prediction of intermediate posteriors
  end
  PostSTDPRed(:,i) = Post_STDPred([Xgrid(:,i),Ygrid(:,i)]);

end


figure
for s=2:j%plot the mean estimate of the unnormalized posteriors for all intermedaite stages, including the final stage
    subplot(ceil((j+1)/2),2,s-1)
    pcolor(Xgrid,Ygrid,PostMeanPred(:,:,s));
    if s==j
        clim([0 max(max(PostRefGrid))])
    end
   colormap(C14)
   colorbar
   shading interp
   hold on
   if s==2
       t0 = plot(Ttrain0,Ltrain0,'Marker','o','LineStyle','none','MarkerEdgeColor',[255 182 30]/255,'MarkerFaceColor',[0.95,0.95,0.95],'MarkerSize',6);
       hold on
   end
   t = plot(Ttrain(NcallEachStage(s-1)+1:NcallEachStage(s),1),Ttrain(NcallEachStage(s-1)+1:NcallEachStage(s),2),'Marker','hexagram','LineStyle','none','MarkerEdgeColor',[255 182 30]/255,'MarkerFaceColor',[0.95,0.95,0.95],'MarkerSize',8);
   xlabel('$\theta_1$','Interpreter','latex')
   ylabel('$\theta_2$','Interpreter','latex')
   title(['$\gamma=$',num2str(gamma_fixed(s),'%.4f')],'Fontsize',12,'Interpreter','latex')
   
end

 subplot(ceil((j+1)/2),2,j)%%Reference
 pcolor(Xgrid,Ygrid,PostRefGrid);
 colormap(C14)
 colorbar
 shading interp
 hold on
 xlabel('$\theta_1$','Interpreter','latex')
 ylabel('$\theta_2$','Interpreter','latex')
 title('Reference','Interpreter','latex','Fontsize',12,'FontName', 'Helvetica')




 %%%%%plot the MCMC samples
 figure
for s=2:j%plot the mean estimate of the unnormalized posteriors for all intermedaite stages, including the final stage
    subplot(ceil((j+1)/2),2,s-1)
    plot(MCSamp(:,1,s), MCSamp(:,2,s),'.');
   xlabel('$\theta_1$','Interpreter','latex')
   ylabel('$\theta_2$','Interpreter','latex')
   title(['$\gamma=$',num2str(gamma_fixed(s),'%.4f')],'Fontsize',12,'Interpreter','latex')
   
end


function PDFMeanPred = LikeFun(z,GPRmodel,gamma)% estimate the posterior mean of the AID indexed by gamma
    [MeanPred,~] = predict(GPRmodel,z);
    PDFMeanPred = exp(gamma*MeanPred);
end

function PDFSTDPred = LikeFunSTD(z,GPRmodel)% estimate the posterior STD of the true likelihood
    [MeanPred,SDPRed] = predict(GPRmodel,z);
    PDFMeanPred = exp(MeanPred+SDPRed.^2/2);%%posterior mean of the likelihood function
    PDFSTDPred = PDFMeanPred.*sqrt((exp(SDPRed.^2)-1));
end

function AcqFvalue = AcqFunTypeI(GPRmodel,gamma,XMC,PriorDen)
%% the acquisition function of old version, developed in Ref. [1], and requiped to specify an extra parameter "beta" 
beta = norminv(0.75);
[MeanPred,STDPred,~] = predict(GPRmodel,XMC);
AcqFvalue = (exp(gamma*(MeanPred+beta*STDPred))-exp(gamma*(MeanPred-beta*STDPred))).*PriorDen(XMC);
end


function PUQValue = PUQFun(GPRmodel,gamma,XMC,PriorPDF,LikeSamp)%%%the PUQ acquisition function
%%%The new Prediction Uncertainty Quantification (PUQ) function, in Ref. [2]
    [MeanPred,SDPRed] = predict(GPRmodel,XMC);
    AIDMeanPRed = exp(gamma*MeanPred+gamma^2*SDPRed.^2/2);
    PUQValue = sqrt(AIDMeanPRed.^2.*(exp(gamma^2*SDPRed.^2)-1)).*PriorPDF(XMC).*LikeSamp;
end


function AcqFvalue = PVCFun(GPRmodel,gamma,XMC,PriorPDF,LikeSamp)%%%the PVC function
%%%%the Posterior Variance Contribution function
    NMC = size(XMC,1);
    [MeanPredSamp,STDPredSamp] = predict(GPRmodel,XMC); 
    [~,CovPred,~] = predictExactWithCov(GPRmodel.Impl,XMC,0.05);
    Term1 = exp(repmat(gamma*MeanPredSamp,1,NMC)+repmat(gamma*MeanPredSamp',NMC,1)+repmat(gamma^2*STDPredSamp.^2/2,1,NMC)+repmat(gamma^2*STDPredSamp'.^2/2,NMC,1));
    AcqFvalue = (mean(Term1.*(exp(gamma^2*CovPred)-1)./repmat(LikeSamp',NMC,1),2)).*PriorPDF(XMC).*LikeSamp;%%V1
    % AcqFvalue = mean(Term1.*(exp(gamma^2*CovPred)-1),2);%V2
end


function AcqFvalue = PLURFun(GPRmodel,gamma,XMC,PriorPDF,LikeSamp)%%the PLUR function
%%%%the Prospective Likelihood Uncertainty Reduction (PEUR) function
    NMC = size(XMC,1);
    [~,CovPred,~] = predictExactWithCov(GPRmodel.Impl,XMC,0.05);
    [MeanPredSamp,STDPredSamp] = predict(GPRmodel,XMC);
    Term1 = exp(repmat(2*gamma*MeanPredSamp'+gamma^2*STDPredSamp'.^2,NMC,1));
    AcqFvalue = mean(Term1.*(exp(gamma^2*CovPred.^2./repmat(STDPredSamp.^2,1,NMC))-1)./repmat(LikeSamp',NMC,1),2).*PriorPDF(XMC);%.*LikeSamp;
end

function AcqFvalue = PEURFun(GPRmodel,gamma,XMC,RandInd,LikeSamp)
%%%%the Prospective Evidence Uncertainty Reduction (PEUR) function
    NMC = size(XMC,1);
    [MeanPredSamp,STDPredSamp] = predict(GPRmodel,XMC);
    MeanPredSamp2 = MeanPredSamp(RandInd,:); 
    STDPredSamp2 = STDPredSamp(RandInd,:); 
    [~,CovPred,~] = predictExactWithCov(GPRmodel.Impl,XMC,0.05);
    CovPred2 = CovPred(:,RandInd);
    Term1 = exp(repmat(gamma*MeanPredSamp'+gamma^2*STDPredSamp'.^2/2+gamma*MeanPredSamp2'+gamma^2*STDPredSamp2'.^2/2,NMC,1))./repmat(LikeSamp'.*LikeSamp(RandInd,:)',NMC,1);
    Term2 = exp(gamma^2*CovPred.*CovPred2./repmat(STDPredSamp.^2,1,NMC))-ones(NMC,NMC);
    AcqFvalue = mean(Term1.*Term2,2);
end

