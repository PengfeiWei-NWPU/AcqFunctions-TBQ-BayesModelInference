
%%%%this code implement the Bayesian Quadrature (BQ), equipped with four
%%%%new acuqisition functions, for estimating posteriors and the associated
%%%%model evidence with desired accuracy. One can refer to Algorithm 1 in
%%%%the following reference for details. Results in Figure 3-6 of this paper are produced with this code 
%[1]. Song, Jingwen, and Wei, Pengfei. "Bayesian Model Inference using Bayesian Quadrature: the Art of Acquisition Functions and Beyond."  
%     arXiv preprint arXiv:2510.08974 (2025), submitted to Mechanical Systems and Signal Processing   
clear; clc
%%Source of examples:Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." International conference on machine learning. PMLR, 2015. 
%%Define the four energy functions
w1 = @(x)sin(2*pi*x(:,1)/4);
w2 = @(x)3*exp(-0.5*((x(:,1)-1)/0.6).^2);
w3 = @(x)3./(1+exp(-(x(:,1)-1)/0.2));
U{1} = @(x) 0.5*((sqrt(sum(x(:,1:2).^2,2))-2)/0.4).^2 - log(exp(-0.5*(x(:,1)-2).^2/0.6^2)+exp(-0.5*(x(:,1)+2).^2/0.6^2));
U{2} = @(x) 0.5*((x(:,2)+w1(x))/0.4).^2;
U{3} = @(x) -log(exp(-0.5*((x(:,2)+w1(x))/0.35).^2)+exp(-0.5*((x(:,2)+w1(x)-w2(x))/0.35).^2));
U{4} = @(x) -log(exp(-0.5*((x(:,2)+w1(x))/0.4).^2)+exp(-0.5*((x(:,2)+w1(x)-w3(x))/0.35).^2));
IndProb = input('Input the index of problem to be solved: ');
AcqFunIndex = input('Input the index of Acqusition function (0. PUQ (old version) 1. PUQ (new version), 2. PVC, 3. PLUR, 4. PEUR):');
%define the likelihood function, proxy function (logarithm of likelihood)
%and prior density
LogLikelihood =@(z)-U{IndProb}(z);
Likelihood = @(z)exp(LogLikelihood(z));
lb = -4;
ub = 4;
nz = 2;
PriorPDF = @(z)unifpdf(z(:,1),-4,4).*unifpdf(z(:,2),-4,4);% Define the prior

%%%compute reference results
Ntest = 1e6;
for i=1:nz
   TestSamp(:,i) = unifrnd(lb,ub,Ntest,1);
end
LikeTest = Likelihood(TestSamp);
ZRef = mean(LikeTest);
ZRefCov = std(LikeTest)/sqrt(Ntest)/ZRef;

%%%setting the algorithm parameters%%%%
N0 = 12; %%Number of initial sample size
NMC = 1e4; % 
StopLimit = 2;
if IndProb == 1|| IndProb ==2%%%parameter settings for diferent examples
   KerFun = 'ardsquaredexponential';%%kernel function for training GP, available: 'ardsquaredexponential',  'ardexponential','ardmatern32','ardmatern52'
   StopThreshold = 0.04;% Stopping threshold
elseif IndProb == 3
   KerFun =  'ardsquaredexponential';%%kernel function for training GP, available: 'ardsquaredexponential',  'ardexponential','ardmatern32','ardmatern52'
   StopThreshold = 0.02;% Stopping threshold
else
   KerFun =  'ardmatern52';%%kernel function for training GP, available: 'ardsquaredexponential',  'ardexponential','ardmatern32','ardmatern52'
   StopThreshold = 0.02;% Stopping threshold
end 
beta = norminv(0.75);% for defining the confidence bounds

%%Initialize the algorithm

Stopflag = 0;
MCSamp = unifinv(lhsdesign(NMC,nz),-4,4);
RandInd = randperm(NMC,NMC);
MCSamp2 = MCSamp(RandInd,:);%%generate the second MC sample matrix by randomly permuting the rows of the first one
Ttrain0 = unifinv(lhsdesign(N0,nz),-4,4);
% Ttrain0 = [1.9941    2.0010    3.4409   -3.3806   -2.7961    0.4029   -2.0037    1.2020   -0.6059   -1.1981    2.9172   -1.6440
%    -2.7129    0.2853    2.3564   -1.4790    2.9661   -2.2535   -0.0818    1.9394   -0.9949   -3.6154    0.7690    3.9410]';
Ytrain0 = LogLikelihood(Ttrain0);
%%%%%%%%%%%%Type I%%%%%%%%%%%%%%%%
Ttrain1 = Ttrain0;
Ytrain1 = Ytrain0;
Ncall1 = N0;
MeanZ = [];%%Record the mean prediction of Z
STDZ_UB = [];%Record the prediction STD of Z
options = optimoptions(@ga,'UseParallel',true,'UseVectorized',true,'PopulationSize',50); % ,'MaxIterations',20,'SwarmSize',20
while 1==1
    GPRmodel1 = fitrgp(Ttrain1,Ytrain1,'KernelFunction',KerFun...
                   ,'BasisFunction','pureQuadratic','Sigma', 1e-3, 'ConstantSigma', true,'SigmaLowerBound', 1e-5);  
    [MeanPredSamp,STDPredSamp,~] = predict(GPRmodel1,MCSamp);
    STDPredSamp = min(STDPredSamp,1e10);
    STDPredSamp = max(STDPredSamp,1e-6);
    SampPred = [MeanPredSamp,STDPredSamp];
    MeanZSamp = exp(MeanPredSamp+STDPredSamp.^2/2);
    % MeanZ(Ncall1-N0+1) = mean(MeanZSamp);%Mean prediction of Z
    % MeanZSamp = exp(MeanPredSamp);
    MeanZ(Ncall1-N0+1) = mean(exp(MeanPredSamp));%Mean prediction of Z
%     STDZ_UB(Ncall1-N0+1) = sqrt(mean((exp(STDPredSamp.^2)-1).*MeanZSamp.^2));%%Std

%%%%%This block is used for compute the covariance of the LGP at MCSamp and MCSamp
       Sigma0 = GPRmodel1.KernelInformation.KernelParameters(end);
       Sigma1 = GPRmodel1.KernelInformation.KernelParameters(1:end-1);
       dis = @(ZN,ZM)pdist2(ZN,ZM,'seuclidean',Sigma1);
       if strcmp(KerFun, 'ardsquaredexponential')==1
           kfcn = @(ZN,ZM)Sigma0^2*exp(-(dis(ZN,ZM).^2)/2);%%kernel function
       elseif strcmp(KerFun, 'ardexponential')==1
          kfcn = @(ZN,ZM)Sigma0^2*exp(-dis(ZN,ZM));
       elseif strcmp(KerFun, 'ardmatern32')==1
           kfcn = @(ZN,ZM)Sigma0^2*(1+sqrt(3)*dis(ZN,ZM)).*exp(-sqrt(3)*dis(ZN,ZM));
       elseif strcmp(KerFun, 'ardmatern52')==1
           kfcn = @(ZN,ZM)Sigma0^2*(1+sqrt(5)*dis(ZN,ZM)+5*dis(ZN,ZM).^2/3).*exp(-sqrt(5)*dis(ZN,ZM));
       end
       K = kfcn(Ttrain1,Ttrain1)+GPRmodel1.Sigma.^2*eye(Ncall1);
       InvK=pinv(K);
       CovPredGP= @(z1,z2)kfcn(z1,z2)-kfcn(z1,Ttrain1)*InvK*kfcn(Ttrain1,z2);%%posterior covariance
       CovValue = diag(CovPredGP(MCSamp,MCSamp2));
       STDZ(Ncall1-N0+1) = sqrt(mean(MeanZSamp.*MeanZSamp(RandInd,:).*(exp(CovValue)-1)));%%Std
%%%%%End
 fprintf('Active Cov of evidence： %.4f\n', STDZ(Ncall1-N0+1)/MeanZ(Ncall1-N0+1));
 fprintf('Mean estimate of evidence： %.4f\n', MeanZ(Ncall1-N0+1));
 fprintf('Number of function calls： %.4f\n', Ncall1);

    if STDZ(Ncall1-N0+1)/MeanZ(Ncall1-N0+1)<StopThreshold
        Stopflag = Stopflag+1;
    else
        Stopflag = 0;
    end
    if Stopflag==StopLimit
        break
    else
        if AcqFunIndex==0
           [Tnew,~] = ga(@(z)PUQFunOld(GPRmodel1,z,beta,PriorPDF),nz,[],[],[],[],-4*ones(1,nz),4*ones(1,nz),[],options);
        elseif AcqFunIndex==1
           [Tnew,~] = ga(@(z)PUQFunNew(GPRmodel1,z,PriorPDF),nz,[],[],[],[],-4*ones(1,nz),4*ones(1,nz),[],options);
        elseif AcqFunIndex==2
           [Tnew,~] = ga(@(z)PVCFun(GPRmodel1,z,MCSamp,CovPredGP,SampPred,PriorPDF),nz,[],[],[],[],-4*ones(1,nz),4*ones(1,nz),[],options);
        elseif AcqFunIndex==3
           [Tnew,~] = ga(@(z)PLURFun(GPRmodel1,z,MCSamp,CovPredGP,SampPred),nz,[],[],[],[],-4*ones(1,nz),4*ones(1,nz),[],options);
        elseif AcqFunIndex==4
           [Tnew,~] = ga(@(z)PEURFun(GPRmodel1,z,MCSamp,CovPredGP,RandInd),nz,[],[],[],[],-4*ones(1,nz),4*ones(1,nz),[],options);
        end
       

       Ncall1 = Ncall1+1;
       Ttrain1 = [Ttrain1;Tnew];
       Ytrain1 = [Ytrain1;LogLikelihood(Tnew)];
    end
end
MeanPosterior = @(z)exp(predict(GPRmodel1,z)).*PriorPDF(z);
% MeanPosterior = @(z)PostMeanLike(GPRmodel1,z).*PriorPDF(z);%/MeanZ(Ncall1-N0+1);%%mean prediction of posetrior density
STDPosterior = @(z)PostSTDLike(GPRmodel1,z).*PriorPDF(z);%/MeanZ(Ncall1-N0+1);%% prediction STD of posetrior density


C1  =[250  218  36; 251  162  56; 233  113  88; 204  71   120;  163  30   154; 111   0     168; 53    4     152]/255;
%%色系1，由浅到深
C2 = [017 050 093; 054 080 131; 115 107 157; 183 131 175; 245 166 115; 252 219 114]/255;
%%色系2，由深到浅
C3 = [084 089 105; 164 117 125; 231 152 124; 139 145 182; 119 113 164]/255;
%%深浅均匀，混彩色
C4 = [045 092 135; 120 153 188; 247 224 165; 247 203 103; 202 139 181; 110 106 145]/255;
%%蓝色和黄色系搭配
C5 = [063 059 114; 097 124 184; 206 217 235; 223 155 146; 192 107 094]/255;
%%由蓝色深到浅再到红色系
C6 = [090 095 163; 143 152 201; 191 192 222; 224 210 229; 231 203 224]/255;
%%蓝色粉色系
C7 = [036 050 087; 041 114 069; 177 196 078; 231 199 054; 237 176 175; 149 052 095; 098 031 101; 141 087 041; 001 138 157]/255;
%%多元炫彩色系
C8 = [051 060 066; 049 102 088; 094 166 156; 194 207 162; 164 121 158; 112 102 144]/255;
%%青紫色系
C9 = [055 127 153; 098 178 182; 242 195 203; 246 166 156; 163 084 083]/255; %适合pcolor
%%小清新色系
C10 =[056 076 159; 094 147 195;157 206 226; 222 242 182;254 249 182;253 204 125;247 135 078;220 059 044;165 000 038]/255; %%适合pcolor
C11 = [012 007 134;076 002 161; 126 003 167;169 035 149;203 071 119;229 108 091; 248 149 064; 253 196 039;239 248 033]/255;%%适合pcolor
C12 = [002 048 071;014 091 118;026 134 163; 070 172 202; 155 207 232;243 249 252;255 202 095;254 168 009;253 152 002;251 132 002]/255;
C13 = [078 098 171; 070 158 180; 135 207 164; 203 233 157;245 251 177;254 232 154;253 185 106;245 117 071;214 064 078; 158 001 066 ]/255;
C14 = [026 049 139;073 108 206;130 170 231;185 210 243; 230 240 254; 249 219 229;247 166 191;228 107 144;192 063 103;154 019 061]/255;
C15 = [000 070 041;012 113 059;055 158 084;119 197 120;186 226 148;236 247 177;254 254 227]/255;
C16 = [090 095 163;143 152 201;191 192 222;224 210 229;231 203 224]/255;



Nplot = 200;
Zplot = linspace(-4,4,Nplot);
[Xgrid,Ygrid] = meshgrid(Zplot,Zplot);
for i=1:Nplot
    Lgrid(:,i) = Likelihood([Xgrid(:,i),Ygrid(:,i)]);
    Postgrid(:,i) = Lgrid(:,i).*unifpdf(Xgrid(:,i),-4,4).*unifpdf(Ygrid(:,i),-4,4);
end
figure
axes('Position',[0.1,0.15,0.35,0.75])
pcolor(Xgrid,Ygrid,Postgrid)
colormap(C14)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('(a). Target posterior','Fontsize',12,'FontName', 'Helvetica')


for i=1:Nplot
    [GPPredMean,GPPRedSTD,~] = predict(GPRmodel1,[Xgrid(:,i),Ygrid(:,i)]);
   MeanPostgrid(:,i) = MeanPosterior([Xgrid(:,i),Ygrid(:,i)]); 
   PredError(:,i) = STDPosterior([Xgrid(:,i),Ygrid(:,i)]);
end
axes('Position',[0.52,0.15,0.37,0.75])
pcolor(Xgrid,Ygrid,MeanPostgrid)
colormap(C14)
colorbar
caxis([0,max(max(Postgrid))])
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
g1 = plot(Ttrain0(1:N0,1),Ttrain0(1:N0,2),'d','LineStyle','none','MarkerSize',8,'MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[241 239 236]/255);
hold on
g2 = plot(Ttrain1(N0+1:end,1),Ttrain1(N0+1:end,2),'p','LineStyle','none','MarkerSize',10,'MarkerEdgeColor',[128 0 32]/255,'MarkerFaceColor',[241 239 236]/255);
title('(b). Mean estimate by Exp-BQ','Fontsize',12,'FontName', 'Helvetica')
legend([g1,g2],{'Initial training points','Actively designed points'},'Orientation','horizontal','Box','off','Fontsize',12,'FontName', 'Helvetica','Location','southoutside');

 fprintf('模型调用次数： %.0f\n', Ncall1);
 fprintf('模型证据估计值： %.4f\n', MeanZ(end));
 fprintf('模型证据估计COV： %.4f\n', STDZ(end)/MeanZ(end));
 fprintf('模型证据参考值： %.4f\n', ZRef);
 fprintf('模型证据参考值CoV： %.4f\n', ZRefCov);



function AcqFvalue = PUQFunOld(GPRmodel,z,beta,PriorPDF)
%% define the non-prospective acquisition function of second type
[MeanPred,STDPred,~] = predict(GPRmodel,z);
AcqFvalue = - (exp(MeanPred+beta*STDPred) - exp(MeanPred-beta*STDPred)).*PriorPDF(z);
end

function AcqFvalue = PUQFunNew(GPRmodel1,z,PriorPDF)
  [MeanPred,STDPred] = predict(GPRmodel1,z);
  STDPred = max(STDPred,1e-6);
  AcqFvalue = -(exp(STDPred.^2)-1).*exp(2*MeanPred+STDPred.^2).*PriorPDF(z);
end

function AcqFvalue = PVCFun(GPRmodel,z,MCSamp,CovPredGP,SampPred,PriorPDF)
    nz = size(z,1);
    NMC = size(MCSamp,1);
    CovPred= CovPredGP(z,MCSamp);%%posterior covariance
%     [MeanPredSamp,STDPredSamp,~] = predict(GPRmodel,MCSamp);
    MeanPredSamp = SampPred(:,1);
    STDPredSamp = SampPred(:,2);
    [MeanPredz,STDPredz,~] = predict(GPRmodel,z);
    STDPredz = max(STDPredz,1e-6);
    STDPredSamp = max(STDPredSamp,1e-6);
    STDPredz = min(STDPredz,1e10);
    STDPredSamp = min(STDPredSamp,1e10);
    Term1 = exp(repmat(MeanPredz,1,NMC)+repmat(MeanPredSamp',nz,1)+repmat(STDPredz.^2/2,1,NMC)+repmat(STDPredSamp'.^2/2,nz,1));
    AcqFvalue = -abs(mean(Term1.*(exp(CovPred)-1),2)).*PriorPDF(z);
end


function AcqFvalue = PLURFun(GPRmodel,z,MCSamp,CovPredGP,SampPred)
    nz = size(z,1);
    NMC = size(MCSamp,1);
    CovPred= CovPredGP(z,MCSamp);%%posterior covariance
    MeanPredSamp = SampPred(:,1);
    STDPredSamp = SampPred(:,2);
    [~,STDPredz,~] = predict(GPRmodel,z);
    STDPredz = max(STDPredz,1e-6);
%     STDPredSamp = max(STDPredSamp,1e-6);
%     STDPredz = min(STDPredz,1e10);
%     STDPredSamp = min(STDPredSamp,1e10);
    Term1 = exp(repmat(2*MeanPredSamp'+STDPredSamp'.^2,nz,1));
    AcqFvalue = -mean(Term1.*(exp(CovPred.^2./repmat(STDPredz.^2,1,NMC))-1),2);
end

function AcqFvalue = PEURFun(GPRmodel,z,MCSamp,CovPredGP,RandInd)
    nz = size(z,1);
    NMC = size(MCSamp,1);
    [MeanPredSamp,STDPredSamp] = predict(GPRmodel,MCSamp);
    MeanPredSamp2 = MeanPredSamp(RandInd,:); 
    STDPredSamp2 = STDPredSamp(RandInd,:); 
    Term1 = exp(repmat(MeanPredSamp'+STDPredSamp'.^2/2+MeanPredSamp2'+STDPredSamp2'.^2/2,nz,1));
    CovPred= CovPredGP(z,MCSamp);%%posterior covariance
    CovPred2= CovPred(:,RandInd);
    [~,STDPredz] = predict(GPRmodel,z);
    STDPredz = max(STDPredz,1e-6);
    Term2 = exp(CovPred.*CovPred2./repmat(STDPredz.^2,1,NMC))-1;
    AcqFvalue = -(mean(Term1.*Term2,2));
end


function MeanPredLike = PostMeanLike(GPRmodel1,z)
  [MeanPred,STDPred] = predict(GPRmodel1,z);
  STDPred = max(STDPred,1e-6);
  MeanPredLike = exp(MeanPred+STDPred.^2/2);
end

function STDPredLike = PostSTDLike(GPRmodel1,z)
  [MeanPred,STDPred] = predict(GPRmodel1,z);
  STDPred = max(STDPred,1e-6);
  STDPredLike = (exp(STDPred.^2)-1).*exp(2*MeanPred+STDPred.^2);
end


