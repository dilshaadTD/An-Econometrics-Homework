clc;clear; 
load data.mat
quan1 = v1(:,1);
qual = v1(:,2);
p1 = v1(:,3);
inc1 = v1(:,4);
mcs = v2(:,1);
p2 = v2(:,2);
inc2 = v2(:,3);
qual0 = v2(:,4);
qual1 = v2(:,5);
N1 = size(quan1,1); 
N2 = size(mcs,1); 
con1=ones(N1,1);
con2=ones(N2,1);
X1 = [con1 qual p1 inc1];

%OLS for starting point
start = inv(X1'*X1)*X1'*quan1;

% Moment condition : mean x(quan1 - expected(quan1))=0 for the first one
%and mean(mcs-expected(mcs)) = 0 for the other
%  Q=objective(beta, quan1, mcs, X1, p2,inc2, qual1, qual0)

% FIRST STAGE ESTIMATION
t=1;                     
betaQN = zeros(4,3);
%Quasi-Newton
opt=optimset('Display', 'iter', 'TolFun', 1e-8, 'TolX', 1e-8, 'MaxIter', 1e+6, 'MaxFunEval', 1e+6);
beta=fminunc('objective', start, opt, quan1, mcs, X1, p2,inc2, qual1, qual0);
betaQN(:,t)=beta;

%SECOND STAGE ESTIMATION
%Update W first, look at objective2.m

%Quasi-Newton
opt=optimset('Display', 'iter', 'TolFun', 1e-8, 'TolX', 1e-8, 'MaxIter', 1e+6, 'MaxFunEval', 1e+6);
beta=fminunc('objective2', start, opt, quan1, mcs, X1, p2,inc2, qual1, qual0,t);
betaQN(:,t+1)=beta;

%ITERATED GMM ESTIMATIONS
% Idea should be this: everytime GMM estimations provide an estimator, we
% should update weights accordingly. 
betaRN=zeros(4,11);
R=1;
betaRN(:,R) = betaQN(:,2);
for R=1:1:10
%Quasi-Newton
opt=optimset('Display', 'iter', 'TolFun', 1e-8, 'TolX', 1e-8, 'MaxIter', 1e+6, 'MaxFunEval', 1e+6);
beta=fminunc('objective3', betaRN(:,R), opt, quan1, mcs, X1, p2,inc2, qual1, qual0,t);
betaRN(:,R+1)=beta;
%Update W, look at objective3.m
end
betaQN(:,t+2)=betaRN(:,R+1);




