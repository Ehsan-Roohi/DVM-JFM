/* Reduced ES-FP: implicit positive Chang-Cooper axial diffusion.
   Axial drift/diffusion are moment-fitted; their finite-grid departures are
   reported, bounded and must vanish under velocity refinement. No clipping. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
static double B(double z){
 if(fabs(z)<1e-5)return 1-z/2+z*z/12;
 if(z>700)return 0;
 if(z< -700)return -z;
 return z/expm1(z);
}
static void matrix(int n,const double*v,double dv,double u,double D,double r,
 double*l,double*d,double*h){
 for(int j=0;j<n;j++){l[j]=h[j]=0;d[j]=1;}
 for(int j=0;j<n-1;j++){
  double z=((v[j]+v[j+1])/2-u)*dv/D;
  double up=r*D/(dv*dv)*B(z),down=r*D/(dv*dv)*B(-z);
  d[j]+=up;d[j+1]+=down;h[j]=-down;l[j+1]=-up;
 }
}
static void solve(int n,const double*l,const double*d,const double*h,
 const double*b,double shift,double*y,double*work){
 double den=d[0]+shift;work[0]=h[0]/den;y[0]=b[0]/den;
 for(int j=1;j<n;j++){
  den=d[j]+shift-l[j]*work[j-1];work[j]=h[j]/den;
  y[j]=(b[j]-l[j]*y[j-1])/den;
 }
 for(int j=n-2;j>=0;j--)y[j]-=work[j]*y[j+1];
}
static void sums(int n,const double*v,const double*g,double dv,double u,
 double*mean,double*variance){
 double a=0,b=0,c=0;
 for(int j=0;j<n;j++){double z=v[j]-u;a+=g[j];b+=z*g[j];c+=z*z*g[j];}
 *mean=b*dv;*variance=c*dv;
}
int collide_serial(int nx,int n,const double*v,double dv,double dt,
 const double*f,double*out,double*diagnostics){
 double *buf=malloc(10*n*sizeof(double));if(!buf)return -1;
 double *l=buf,*d=l+n,*h=d+n,*w=h+n,*g=w+n,*gu=g+n,*gd=gu+n,
 *rhs=gd+n,*hh=rhs+n,*kk=hh+n;
 double maxfit=0,maxu=0,maxD=0,minPi=1e300,maxinv=0;
 for(int i=0;i<nx;i++){
  const double*oldg=f+i*n,*oldh=f+nx*n+i*n,*oldk=f+2*nx*n+i*n;
  double rho=0,mom=0,e=0;
  for(int j=0;j<n;j++){rho+=oldg[j]*dv;mom+=oldg[j]*v[j]*dv;e+=(oldg[j]*v[j]*v[j]+oldh[j])*dv;}
  double u=mom/rho,T=(e/rho-u*u)/3,xx=0;
  for(int j=0;j<n;j++)xx+=oldg[j]*(v[j]-u)*(v[j]-u)*dv;
  double sig=xx-rho*T,anis=fmax(sig/rho,-sig/(2*rho));
  double beta=-1.25;
  /* Fei Eq25: beta=max(-5/4,-T/(lambda_max-T)). The 1e-12
     inward relative margin makes semidefiniteness strictly positive. */
  if(anis>0)beta=fmax(beta,-(1.0-1e-12)*T/anis);
  double tau=2*(1-beta)/rho,r=dt/tau;
  double preDx=T+beta*sig/rho,preDy=T-beta*sig/(2*rho);
  if(!(rho>0&&T>0&&preDx>0&&preDy>0&&isfinite(e))){free(buf);return 10+i;}
  double signew=sig/(1+2*(1-beta)*r),target=rho*T+signew;
  double physicalD=T+beta*signew/rho,Py=T-beta*signew/(2*rho);
  if(preDx<minPi)minPi=preDx;
  if(preDy<minPi)minPi=preDy;
  /* Fixed physical diffusion: no fitted drift or diffusion coefficient.
     Positive exponential minimum-relative-entropy projection enforces the
     BE mass, mean and axial variance. Its size is measured and bounded. */
  double ue=u,D=physicalD;
  matrix(n,v,dv,ue,D,r,l,d,h);solve(n,l,d,h,oldg,0,g,w);
  memcpy(gu,g,n*sizeof(double));
  double aa=0,bb=0,scale=sqrt(T),targetv=target/(rho*T);int good=0;
  for(int it=0;it<30;it++){
   double z0=0,z1=0,z2=0,z3=0,z4=0;
   for(int j=0;j<n;j++){double z=(v[j]-u)/scale,ex=aa*z+bb*z*z;if(!isfinite(ex)||fabs(ex)>600){free(buf);return 6000+i;}g[j]=gu[j]*exp(ex);double q=g[j]*dv;z0+=q;z1+=q*z;z2+=q*z*z;z3+=q*z*z*z;z4+=q*z*z*z*z;}
   if(!(z0>0&&isfinite(z0))){free(buf);return 6000+i;}
   double m1=z1/z0,m2=z2/z0,m3=z3/z0,m4=z4/z0;
   double err=fmax(fabs(m1),fabs(m2-targetv));
   if(err<2e-13){for(int j=0;j<n;j++)g[j]*=rho/z0;good=1;if(err>maxfit)maxfit=err;break;}
   double A=m2-m1*m1,C=m3-m1*m2,E=m4-m2*m2,det=A*E-C*C;
   if(!(det>1e-24&&isfinite(det))){free(buf);return 6000+i;}
   double da=(-m1*E+C*(m2-targetv))/det,db=(-A*(m2-targetv)+C*m1)/det;
   double step=1;if(fabs(da)+fabs(db)>.25)step=.25/(fabs(da)+fabs(db));
   aa+=step*da;bb+=step*db;
  }
  if(!good){free(buf);return 7000+i;}
  double corr=0,corr2=0;
  for(int j=0;j<n;j++){double z=(v[j]-u)/scale;corr+=fabs(g[j]-gu[j])*dv;corr2+=fabs(g[j]-gu[j])*z*z*dv;}
  double correction=fmax(corr/rho,corr2/rho);
  if(!(correction<.02&&isfinite(correction))){free(buf);return 8000+i;}
  if(correction>maxD)maxD=correction;
  for(int j=0;j<n;j++)rhs[j]=oldh[j]+4*r*Py*g[j];
  solve(n,l,d,h,rhs,2*r,hh,w);
  for(int j=0;j<n;j++)rhs[j]=oldk[j]+16*r*Py*hh[j];
  solve(n,l,d,h,rhs,4*r,kk,w);
  double rn=0,mn=0,en=0;
  for(int j=0;j<n;j++){
   if(!(g[j]>=0&&hh[j]>=0&&kk[j]>=0&&isfinite(g[j]+hh[j]+kk[j]))){free(buf);return 4000+i;}
   if(g[j]*kk[j]<hh[j]*hh[j]*(1-1e-12)){free(buf);return 9000+i;}
   out[i*n+j]=g[j];out[nx*n+i*n+j]=hh[j];out[2*nx*n+i*n+j]=kk[j];
   rn+=g[j]*dv;mn+=g[j]*v[j]*dv;en+=(g[j]*v[j]*v[j]+hh[j])*dv;
  }
  double ie=fmax(fabs(rn-rho)/rho,fmax(fabs(mn-mom)/fmax(1,fabs(mom)),fabs(en-e)/fmax(1,fabs(e))));
  if(ie>maxinv)maxinv=ie;
  if(fabs(ue-u)>maxu)maxu=fabs(ue-u);
  /* diagnostics[2] bounds positive projection, D/physicalD is identically one. */
  if(ie>2e-11){free(buf);return 5000+i;}
 }
 diagnostics[0]=maxfit;diagnostics[1]=maxu;diagnostics[2]=maxD;diagnostics[3]=minPi;diagnostics[4]=maxinv;
 free(buf);return 0;
}

#include <omp.h>
int collide(int nx,int n,const double*v,double dv,double dt,const double*f,double*out,double*diagnostics){
 int failed=0;
 for(int k=0;k<5;k++)diagnostics[k]=(k==3?1e300:0);
 #pragma omp parallel
 {
  double *input=malloc(3*n*sizeof(double)),*output=malloc(3*n*sizeof(double));
  if(!input||!output){
   #pragma omp critical
   {failed=-1;}
  }
  #pragma omp for schedule(static)
  for(int i=0;i<nx;i++){
   if(!input||!output)continue;
   for(int k=0;k<3;k++)memcpy(input+k*n,f+k*nx*n+i*n,n*sizeof(double));
   double di[5]={0,0,0,0,0};int code=collide_serial(1,n,v,dv,dt,input,output,di);
   if(code){
    #pragma omp critical
    {if(code!=2000)failed=code;else if(failed==0)failed=code;}
   }else{
    for(int k=0;k<3;k++)memcpy(out+k*nx*n+i*n,output+k*n,n*sizeof(double));
    #pragma omp critical
    {for(int k=0;k<5;k++){if(k==3){if(di[k]<diagnostics[k])diagnostics[k]=di[k];}else if(di[k]>diagnostics[k])diagnostics[k]=di[k];}}
   }
  }
  free(input);free(output);
 }
 return failed;
}
