clc; clear all; close all;

% Loading the data.
load('trainingdata.mat')
load('testingdata.mat')

%% Setting the parameters
Ntrain = size(X,2);
% Formation of the kernel matrix (PhiX) from anchors and X.
% Get anchors
n_anchors = 1000; rand('seed',1);
% Anchors are randomly sampled from the training dataset.
anchor = X(:,randsample(Ntrain,n_anchors));
% Set the experimental parameters as given in the paper
sigma = 1; 
lambda = 1e-2; 
alpha = 1e-5;
beta = 1e-4;
gamma = 1e-6;

%% Phi of training data.
Dis = EuDist2(X',anchor',0);
bandwidth = mean(mean(Dis)).^0.5;
clear Dis;
PhiX = exp(-sqdist(X',anchor')/(bandwidth*bandwidth));
PhiX = [PhiX, ones(Ntrain,1)];
PhiX = PhiX';

%% Creating the Laplacian Matrix
N_K = 10; % k-nearest neighbors
eu_dist = sqdist(X',X') + 1e10*eye(Ntrain);  %Euclidian distance of each element from other.
fprintf('eu dist done\n');
A = zeros(Ntrain,Ntrain);
% eu_dist will be symmetric. Check in each row to apply affinity formula.
for i = 1 : Ntrain
    % Sort ith row and get sorted indices
    [~, ind] = sort(eu_dist(i, :) );
    % Affinities to N_K nearest neighbors
    for j = 1 : N_K
        A(i, ind(j)) = exp( -1*(eu_dist(i, ind(j)))/(sigma*sigma) );
        A(ind(j), i) = A(i, ind(j));
    end
end
clear eudist;
D=diag(sum(A,2));
L = D-A;
clear A D

%% Optimization Parameters
debug = 1;
tol = 1e-5;
nbits = 64; % number of bits of the hash codes.
% Init B
randn('seed',3);
B = sign(randn(Ntrain,nbits));
B = B';
c = size(Y,1); % size of the semantic alignment matrix
W = randn(nbits,c);

%% Optimization

% R-Step
R = randn(c,c); R = orth(R); %Initialize R
opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
[R, out]= OptStiefelGBB(R, @fun, opts, Y, W, B);

% W-Step
W = (B*B' + lambda*eye(size(B,1)))\(B*Y'*R);

% Updating P.
P = (PhiX*PhiX' + (beta/alpha)*eye(n_anchors+1) +  (gamma/alpha)*PhiX*L*PhiX')\(PhiX*B');

% B-Step
maxItr = 10;

for i = 1: maxItr    
    if debug,fprintf('Iteration  %03d: ',i);end
    % B-step
    H = W*R'*Y + alpha*P'*PhiX;  
    B = zeros(size(B));
    for time = 1:10           
       Z0 = B;
       for k = 1 : size(B,1)
             Zk = B; Zk(k,:) = [];
             Wkk = W(k,:); Wk = W; Wk(k,:) = [];                    
             B(k,:) = sign(H(k,:) -  Wkk*Wk'*Zk);
       end
            
       if norm((B-Z0),'fro') < 1e-6 * norm(Z0,'fro')
              break
       end
    end
    
    % R-Step Again
    [R, out]= OptStiefelGBB(R, @fun, opts, Y, W, B);    
    
    % W-Update Again
    W = (B*B' + lambda*eye(size(B,1)))\(B*Y'*R);
    
    P0 = P;
    % P-step Again
    P = (PhiX*PhiX' + (beta/alpha)*eye(n_anchors+1) +  (gamma/alpha)*PhiX*L*PhiX')\(PhiX*B');
    bias = norm(B - P'*PhiX,'fro');
    if debug, fprintf('  bias=%g\n',bias); end
    
    if bias < tol*norm(B,'fro')
            break;
    end 
    if norm(P-P0,'fro') < tol * norm(P0)
        break;
    end

end

%%
% Phi of testing data.
Phi_Xtest = exp(-sqdist(Xtest',anchor')/(2*sigma*sigma));
Phi_Xtest = [Phi_Xtest, ones(size(Phi_Xtest,1),1)];
Phi_Xtest = Phi_Xtest';

%%
% Evaluation
display('Evaluation starts ...');
AsymDist = 0; % Use asymmetric hashing or not
if AsymDist 
    H = H > 0; % directly use the learned bits for training data
else
    H = P'*PhiX > 0;
end
tH = P'*Phi_Xtest > 0;
HH = [H';tH'];
HH(HH<0) = 0;
HH = logical(HH);
dataset.label = [trainlabel;testlabel];
dataset.neighborType='label';
dd = randperm(size(HH,1));
ff = dd(1:2000);
gg = dd(2001:size(HH,1));
dataset.indexTest = ff;
dataset.indexTrain = gg;
Qlabel = dataset.label(ff,:);
Dlabel = dataset.label(gg,:);
cateTrainTest=calcNeighbor(dataset,dataset.indexTest,dataset.indexTrain)';
Btest = HH(ff,:);
Btrain = HH(gg,:);
B = compactbit(Btrain);
tB = compactbit(Btest);
hammTrainTest = hammingDist(tB, B)';
hammRadius = 2;
Ret = (hammTrainTest <= hammRadius+0.00001);
% get hash lookup using hamming ball
[cateP, cateR] = evaluate_macro(cateTrainTest, Ret); % category
cateF1 = F1_measure(cateP, cateR);
%clear cateP cateR
% get hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest,1);
[cateMAP] = cat_apcal(Dlabel,Qlabel,HammingRank);