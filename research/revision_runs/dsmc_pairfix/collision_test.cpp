#define main solver_main
#include "corrected.cpp"
#undef main
int main(){
 // Every distinct pair has exactly one preimage (ia, reduced ib).
 for(int n: {2,3,8,60}){vector<int> counts(n*n,0);for(int ia=0;ia<n;ia++)for(int b=0;b<n-1;b++){int ib=b;if(ib>=ia)++ib;counts[ia*n+ib]++;}for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(counts[i*n+j]!=(i!=j))throw runtime_error("pair bijection");}
 // N=2, one candidate, g=gmax: must collide, conserve momentum/energy.
 for(int seed=1;seed<=10000;seed++){
 Args a;a.cells=1;a.dt=1;a.sigmaT=1;
 Particles p;p.push(.5,1,0,0);p.push(.5,-1,0,0);
 vector<vector<int>> bins{{0,1}};vector<double> gm{2},rem{0};ProfileAccum acc(1);mt19937_64 rng(seed);
 collide_ntc(p,bins,gm,rem,acc,true,a,0,1,.5,rng);
 if(acc.prod_coll[0]!=1)throw runtime_error("lost candidate");
 double e=0;for(int k=0;k<2;k++)e+=p.vx[k]*p.vx[k]+p.vy[k]*p.vy[k]+p.vz[k]*p.vz[k];
 if(abs(e-2)>1e-12 || abs(p.vx[0]+p.vx[1])+abs(p.vy[0]+p.vy[1])+abs(p.vz[0]+p.vz[1])>1e-12)throw runtime_error("conservation");
 }
 cout<<"PASS bijection, 10000 exact candidate and conservation trials\n";
}