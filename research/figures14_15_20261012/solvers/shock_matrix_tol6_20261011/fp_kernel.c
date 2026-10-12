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
int collide(int nx,int n,const double*v,double dv,double dt,
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
  double sig=xx-rho*T,beta=-1.25,tau=4.5/rho,r=dt/tau;
  double preDx=T+beta*sig/rho,preDy=T-beta*sig/(2*rho);
  if(!(rho>0&&T>0&&preDx>0&&preDy>0&&isfinite(e))){free(buf);return 10+i;}
  double signew=sig/(1+4.5*r),target=rho*T+signew;
  double physicalD=T+beta*signew/rho,Py=T-beta*signew/(2*rho);
  if(preDx<minPi)minPi=preDx;
  if(preDy<minPi)minPi=preDy;
  double ue=u,D=physicalD,em=0,ev=0;int good=0;
  for(int it=0;it<8;it++){
   matrix(n,v,dv,ue,D,r,l,d,h);solve(n,l,d,h,oldg,0,g,w);
   sums(n,v,g,dv,u,&em,&ev);double err=fmax(fabs(em)/(rho*sqrt(T)),fabs(ev-target)/(rho*T));
   if(err<2e-13){good=1;if(err>maxfit)maxfit=err;break;}
   double hu=1e-4*fmax(1,sqrt(T)),hd=1e-4*D,um,uv,dm,dv2;
   matrix(n,v,dv,ue+hu,D,r,l,d,h);solve(n,l,d,h,oldg,0,gu,w);sums(n,v,gu,dv,u,&um,&uv);
   matrix(n,v,dv,ue,D+hd,r,l,d,h);solve(n,l,d,h,oldg,0,gd,w);sums(n,v,gd,dv,u,&dm,&dv2);
   double a=(um-em)/hu,b=(dm-em)/hd,c=(uv-ev)/hu,dd=(dv2-ev)/hd,det=a*dd-b*c;
   if(!isfinite(det)||fabs(det)<1e-24){free(buf);return 1000+i;}
   double du=(-em*dd+b*(ev-target))/det,dD=(-a*(ev-target)+c*em)/det;
   ue+=du;D+=dD;
   if(!(D>0&&isfinite(D)&&fabs(ue-u)<dv&&fabs(D/physicalD-1)<.02)){free(buf);return 2000+i;}
  }
  if(!good){free(buf);return 3000+i;}
  matrix(n,v,dv,ue,D,r,l,d,h);
  for(int j=0;j<n;j++)rhs[j]=oldh[j]+4*r*Py*g[j];
  solve(n,l,d,h,rhs,2*r,hh,w);
  for(int j=0;j<n;j++)rhs[j]=oldk[j]+16*r*Py*hh[j];
  solve(n,l,d,h,rhs,4*r,kk,w);
  double rn=0,mn=0,en=0;
  for(int j=0;j<n;j++){
   if(!(g[j]>=0&&hh[j]>=0&&kk[j]>=0&&isfinite(g[j]+hh[j]+kk[j]))){free(buf);return 4000+i;}
   out[i*n+j]=g[j];out[nx*n+i*n+j]=hh[j];out[2*nx*n+i*n+j]=kk[j];
   rn+=g[j]*dv;mn+=g[j]*v[j]*dv;en+=(g[j]*v[j]*v[j]+hh[j])*dv;
  }
  double ie=fmax(fabs(rn-rho)/rho,fmax(fabs(mn-mom)/fmax(1,fabs(mom)),fabs(en-e)/fmax(1,fabs(e))));
  if(ie>maxinv)maxinv=ie;
  if(fabs(ue-u)>maxu)maxu=fabs(ue-u);
  if(fabs(D/physicalD-1)>maxD)maxD=fabs(D/physicalD-1);
  if(ie>2e-11){free(buf);return 5000+i;}
 }
 diagnostics[0]=maxfit;diagnostics[1]=maxu;diagnostics[2]=maxD;diagnostics[3]=minPi;diagnostics[4]=maxinv;
 free(buf);return 0;
}
