

addpath('../../../../Code+data/Utilities')



%getting the data
[a,b,c]=xlsread('game_one_37.csv');
x_loc=a(:,4); y_loc=a(:,5);
x_tr = real(log(x_loc./(94 - x_loc))); y_tr = real(log(y_loc./(50 - y_loc)));
loc = [x_tr, y_tr]';
Y=diff(loc, 1, 2); Y=Y(:,2:end);  [q, T] = size(Y);

%plot of the differenced data
plot(Y'); xlim([0,T]);  ylim([-0.5,0.5]); 
legend('x', 'y'); title({'Differenced Location Data (Logistic Transformed)', 'Segment 2'});

% DLM 
F = [1; 0]; G=[1 1; 0 1]; p=length(F); 
alpha=0.90; % power discount factor for marginal likelihoods 

% discount factors 
d_delt = [0.8:0.01:1]; d_bet = [0.8:0.01:1];      % range of standards 
ndelta = length(d_delt); nbeta = length(d_bet); 
mlik=zeros(ndelta,nbeta); % 2D array to save log likelihood function for discounts
ilik=floor(T/10); % only save mlik contributions after this time point
nint=zeros(ndelta,nbeta); % 2D array to save % times of interventions

% intervention discounts 
dropped_d = 0.9; dropped_b=0.1; 
yeps = 0.01;   % tail error prob on forecast errors to reject as intervention

% general storage arrays 
z = zeros(p,q);   zq=zeros(q,1);  

%priors
p=2; n0=3; %low df so will adapt quickly 
h0=n0+q-1;  D0=h0*eye(q)*0.01; % multiplying by smaller value says smaller percentage of data is noise 
M0=z; C0=100*eye(p);                       % initial Theta prior 
n = n0; h=h0; D = D0;                      % initial Sigma prior

% iterate over models defined by pairs of (delta,beta)  
for id = 1:ndelta 
    delta = d_delt(id); 
    for ib = 1:nbeta
        beta = d_bet(ib);  
                   % reset priors at t=0 for new model: 
        Mt = M0; Ct=C0;        % initial Theta prior 
        n = n0; h=h0; D = D0;  St=D/h;         % initial Sigma prior
            
        % FF ... .
        for t = 1:T
       
            %  evolutions using standard discounts 
            hold=h; nold=n; Dold=D; % save priors in case of need to use alt discounts
            h  = beta*h;  n=h-q+1;  D = beta*D;  snt(t)=n;             
            Mt = G*Mt'; ft = Mt'*F; 
            Rt = G*Ct*G'/delta; qvt = 1 + F'*Rt*F;  
            et = Y(:,t) - ft; set= et./sqrt(qvt*diag(St)); % one step ahead forecast 
            %checking the size of the errors
            tt = tcdf(set, n);
            
            % check for wild obsn and intervene if flagged ...
            if (tt(1,:)<yeps || tt(1,:)>1-yeps || tt(2,:)<yeps || tt(2,:)>1-yeps)
                interventiont=1; 
                h  = max(q,dropped_b*hold);  n=h-q+1;  D = dropped_b*Dold;   
                Ct = G*Ct*G'/dropped_d;  Ct=(Ct+Ct')/2; 
                if (t>ilik), 
                    nint(id,ib) = nint(id,ib) + 1;   
                end
            else
                interventiont=0;
                if (t>ilik), 
                    
                    %ltpdf function
                    x = et; mm = zq; w = qvt;
                    qq=size(x,1); C=chol(D)'; e=inv(C)*(x-mm);  d=n+qq-1; 
                    ltpdf = qq*log(2)/2 - qq*log(2*pi*w)/2 -sum(log(diag(C))) -(d+1)*log(1+(e'*e)/w)/2;
                    ltpdf = ltpdf + sum(gammaln((1+d-(0:qq-1))/2) - gammaln((d-(0:qq-1))/2)); 
                    
                    mlik(id,ib) = alpha*mlik(id,ib) + ltpdf;   
                end
                At = Rt*F/qvt;  h=h+1; n=n+1; D = D+et*et'/qvt;  D=(D+D')/2;
                Mt = Mt + At*et'; Ct = Rt - At*At'*qvt;   Rt=(Rt+Rt')/2; 
            end
            
            St=D/h;  
  
        end
    end
end

nint=nint*100/(T-ilik);
 
 

%evaluating the likelihood/posterior
mlik=exp(mlik-max(max(mlik)));  plik=mlik/sum(sum(mlik));  % 2-D array of joint post probs
	 pdelta=sum(plik,2)';			% marginal post for delta
		subplot(2,1,1); bar(d_delt,pdelta); colormap white; title('Posterior for \delta')	
        xlim([min(d_delt) max(d_delt)]); box off; xlabel('\delta')
        pbeta=sum(plik,1)				% marginal post for beta
		subplot(2,1,2); bar(d_bet,pbeta); colormap white; title('Posterior for \beta')
        xlim([min(d_delt) max(d_delt)]); box off; xlabel('\beta')

print -djpeg -r300 formegan.jpg






        