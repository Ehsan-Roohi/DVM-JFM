#include <math.h>
#include <stddef.h>
/* Same upwind interfaces and exact frozen-moment BGK exponential as NumPy.
   No fast-math, clipping, target normalization or changed transport law. */
int step_bgk(int nx,int nv,const double *f,const double *v,const double *w,
             const double *left,const double *right,double dt,double dx,double *out,
             double *maxdefect){
 size_t block=(size_t)nx*nv; *maxdefect=0;
 for(int i=0;i<nx;i++){
  double rho=0,mom=0,e2=0;
  for(int j=0;j<nv;j++){
   for(int k=0;k<3;k++){
    size_t q=k*block+(size_t)i*nv+j; double fl,fr;
    if(v[j]>=0){fl=i?f[q-nv]:left[k*nv+j];fr=f[q];}
    else{fl=f[q];fr=i+1<nx?f[q+nv]:right[k*nv+j];}
    double a=f[q]-(dt/dx)*v[j]*(fr-fl);
    if(!isfinite(a)||a<0)return 1;
    out[q]=a;
   }
   double g=out[(size_t)i*nv+j],h=out[block+(size_t)i*nv+j];
   rho+=g*w[j];mom+=g*w[j]*v[j];e2+=(g*v[j]*v[j]+h)*w[j];
  }
  double u=mom/rho,T=(e2/rho-u*u)/3.;
  if(!isfinite(rho)||!isfinite(T)||rho<=0||T<=0)return 2;
  double decay=exp(-dt*rho),factor=rho/sqrt(2*acos(-1.)*T);
  double r1=0,m1=0,e1=0;
  for(int j=0;j<nv;j++){
   double c=v[j]-u,G=factor*exp(-c*c/(2*T));
   double targets[3]={G,2*T*G,8*T*T*G};
   for(int k=0;k<3;k++){
    size_t q=k*block+(size_t)i*nv+j;
    out[q]=targets[k]+(out[q]-targets[k])*decay;
    if(!isfinite(out[q])||out[q]<0)return 3;
   }
   double g=out[(size_t)i*nv+j],h=out[block+(size_t)i*nv+j];
   r1+=g*w[j];m1+=g*w[j]*v[j];e1+=(g*v[j]*v[j]+h)*w[j];
  }
  double d=fmax(fabs(r1-rho)/fmax(fabs(rho),1.),
           fmax(fabs(m1-mom)/fmax(fabs(mom),1.),fabs(e1-e2)/fmax(fabs(e2),1.)));
  if(d>*maxdefect)*maxdefect=d;
  if(d>5e-11)return 4;
 }
 return 0;
}
