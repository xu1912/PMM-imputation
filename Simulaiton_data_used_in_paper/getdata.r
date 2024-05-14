library(sampling)
library(MASS)

gedata<-function(B,N,n,id_m,id_resp,id_design,rou){
###covariates and random error:

mu0<-rep(0,4)
Sigma0<-diag(rep(1-rou,4))
Sigma0<-Sigma0+rou
X<-mvrnorm(B*N,mu=mu0,Sigma=Sigma0)

x1<-X[,1]
x2<-X[,2]
x3<-X[,3]
x4<-X[,4]
epsilon<-rnorm(B*N)

###outcome regression model:
if(id_m==1){
        y<-1+x1+x2+x3+x4+epsilon
}
else if(id_m==2){
        y <- 1 + x1^2 + x4*x3 + x1*x2 + x3^3 + x4^4 + epsilon
}else if(id_m==3){
    range_min <- -2
    range_max <- 2

    beta_0_thru_3 <- runif(4,range_min,range_max)

    alpha_1.1 <- runif(5,range_min,range_max)
    alpha_2.1 <- runif(5,range_min,range_max)
    alpha_3.1 <- runif(5,range_min,range_max)

    # First layer
    a_1.1 <- log( 1 + exp( alpha_1.1[1] + alpha_1.1[2]*x1 + alpha_1.1[3]*x2 + alpha_1.1[4]*x3 + alpha_1.1[5]*x4 ) ) + rnorm(B*N)
    a_2.1 <- log( 1 + exp( alpha_2.1[1] + alpha_2.1[2]*x1 + alpha_2.1[3]*x2 + alpha_2.1[4]*x3 + alpha_2.1[5]*x4 ) ) + rnorm(B*N)
    a_3.1 <- log( 1 + exp( alpha_3.1[1] + alpha_3.1[2]*x1 + alpha_3.1[3]*x2 + alpha_3.1[4]*x3 + alpha_3.1[5]*x4 ) ) + rnorm(B*N)

    alpha_1.2 <- runif(4,range_min,range_max)
    alpha_2.2 <- runif(4,range_min,range_max)
    alpha_3.2 <- runif(4,range_min,range_max)

    a_1.2 <- log( 1 + exp( alpha_1.2[1] + alpha_1.2[2]*a_1.1 + alpha_1.2[3]*a_2.1 + alpha_1.2[4]*a_3.1 ) ) + rnorm(B*N)
    a_2.2 <- log( 1 + exp( alpha_2.2[1] + alpha_2.2[2]*a_1.1 + alpha_2.2[3]*a_2.1 + alpha_2.2[4]*a_3.1 ) ) + rnorm(B*N)
    a_3.2 <- log( 1 + exp( alpha_3.2[1] + alpha_3.2[2]*a_1.1 + alpha_3.2[3]*a_2.1 + alpha_3.2[4]*a_3.1 ) ) + rnorm(B*N)

    # y = third layer
    y <- beta_0_thru_3[1] + beta_0_thru_3[2]*a_1.2 + beta_0_thru_3[3]*a_2.2 + beta_0_thru_3[4]*a_3.2 + epsilon

}

###nonresponse model:
if(id_resp==1){
    alpha<-c(-0.36,1,1,1,1)
}
else if(id_resp==2){
    alpha<-c(1.38,1,1,1,1)
}
p<-0.1+0.9*exp(alpha[1]+alpha[2]*x1+alpha[3]*x2+alpha[4]*x3+alpha[5]*x4)/(1+exp(alpha[1]+alpha[2]*x1+alpha[3]*x2+alpha[4]*x3+alpha[5]*x4))
r<-rbinom(B*N,1,p)


###sampling design:

if(id_design==1){

    sI<-NULL
    iter<-0
    repeat{
        iter<-iter+1
        sI0<-rep(0,N)
        id_s<-sample(1:N,n)
        sI0[id_s]<-1
        sI<-c(sI,sI0)
    if(iter==B){break}
    }
    w<-rep(N/n,B*N)
    sy<-y[sI==1]
    sr<-r[sI==1]
    sx1<-x1[sI==1]
    sx2<-x2[sI==1]
    sx3<-x3[sI==1]
    sx4<-x4[sI==1]
    sw<-w[sI==1]
}

###randomized systematic sampling:
if(id_design==2){

    v<-rnorm(B*N)
    s<-log(0.1*abs(y+v)+4)

    fPi<-function(zz){
        res_Pi<-n*zz/sum(zz)
        return(res_Pi)
    }

    Ms<-matrix(s,B,N,byrow=T)
    Ms2<-apply(Ms,1,fPi)
    Pi<-as.numeric(Ms2)
    Ms3<-apply(Ms2,2,UPrandomsystematic)
    sI<-as.numeric(Ms3)
    w<-1/Pi

    sy<-y[sI==1]
    sr<-r[sI==1]
    sx1<-x1[sI==1]
    sx2<-x2[sI==1]
    sx3<-x3[sI==1]
    sx4<-x4[sI==1]
    sw<-w[sI==1]
}

My<-matrix(sy,B,n,byrow=T)
Mr<-matrix(sr,B,n,byrow=T)
Mx1<-matrix(sx1,B,n,byrow=T)
Mx2<-matrix(sx2,B,n,byrow=T)
Mx3<-matrix(sx3,B,n,byrow=T)
Mx4<-matrix(sx4,B,n,byrow=T)
Mw<-matrix(sw,B,n,byrow=T)

M<-cbind(My,Mr,Mx1,Mx2,Mx3,Mx4,Mw)
M<-t(M)
aM<-as.numeric(M)
Res<-array(aM,c(n,7,B))

theta0=mean(y)
res<-list(Res,theta0)
return(res)
}
