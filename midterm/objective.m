function Q=objective(beta, quan1, mcs, X1, p2,inc2, qual1, qual0)
N1 = size(quan1,1); 
N2 = size(mcs,1); 
f=X1'*(quan1-X1*beta);
a1 = beta(1)+beta(4)*inc2+beta(3)*qual1;
a0 = beta(1)+beta(4)*inc2+beta(3)*qual0;
h=mcs+(1/2*beta(2))*(a1.^2-a0.^2)+beta(3)*p2'*(qual1-qual0);
g1 = (1/N1)*sum(f);
g2 = (1/N2)*sum(h);
W1 = eye;
W2 = eye;
Q =(N1/(N1+N2))*g1*W1*g1+(N2/(N1+N2))*g2*W2*g2;

