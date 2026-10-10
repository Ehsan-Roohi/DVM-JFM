#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <sstream>
#include <stdexcept>

using namespace std;

static const double PI = 3.1415926535897932384626433832795;

struct Args {
    int cells = 300;
    int steps = 100000;
    int sample_start = 30000;
    int sample_every = 10;
    int out_every = 20000;
    int seed = 1;

    double M1 = 3.0;
    double gamma = 5.0/3.0;
    double L = 120.0;
    double dt = 0.005;
    double ppc = 80.0;

    // Total collision cross section. With n1=1 and sigmaT=1/sqrt(2),
    // upstream mean free path is approximately 1.
    double sigmaT = 1.0/std::sqrt(2.0);

    // Initial upper bound for relative speed in NTC.
    double gmax_factor = 10.0;

    string prefix = "M3_dsmc_c4";
};

struct GasState {
    double n;
    double u;
    double T;
};

struct Particles {
    vector<double> x, vx, vy, vz;

    void reserve(size_t n) {
        x.reserve(n);
        vx.reserve(n);
        vy.reserve(n);
        vz.reserve(n);
    }

    size_t size() const { return x.size(); }

    void push(double xx, double ux, double uy, double uz) {
        x.push_back(xx);
        vx.push_back(ux);
        vy.push_back(uy);
        vz.push_back(uz);
    }

    void resize(size_t n) {
        x.resize(n);
        vx.resize(n);
        vy.resize(n);
        vz.resize(n);
    }
};

struct ProfileAccum {
    int cells;
    long long nsamp = 0;

    vector<double> n, ux, uy, uz, T;
    vector<double> qx, sigxx, c4, Delta, Rxx, S;

    // Extra fields for scalar-excess budget:
    // JDelta = int v_x |c|^4 f dv
    // KDelta = int v_x |c|^2 c_x f dv
    vector<double> JDelta, KDelta;

    // Direct collision production of int |c|^4 f dv.
    vector<double> prod_c4, prod_coll;

    ProfileAccum(int nc=0) { reset(nc); }

    void reset(int nc) {
        cells = nc;
        nsamp = 0;

        n.assign(nc,0.0);
        ux.assign(nc,0.0);
        uy.assign(nc,0.0);
        uz.assign(nc,0.0);
        T.assign(nc,0.0);

        qx.assign(nc,0.0);
        sigxx.assign(nc,0.0);
        c4.assign(nc,0.0);
        Delta.assign(nc,0.0);
        Rxx.assign(nc,0.0);
        S.assign(nc,0.0);

        JDelta.assign(nc,0.0);
        KDelta.assign(nc,0.0);

        prod_c4.assign(nc,0.0);
        prod_coll.assign(nc,0.0);
    }
};

static string get_arg(int argc, char** argv, const string& key, const string& def) {
    for (int i=1; i<argc-1; i++) {
        if (string(argv[i]) == key) return string(argv[i+1]);
    }
    return def;
}

static int get_arg_int(int argc, char** argv, const string& key, int def) {
    return stoi(get_arg(argc, argv, key, to_string(def)));
}

static double get_arg_double(int argc, char** argv, const string& key, double def) {
    return stod(get_arg(argc, argv, key, to_string(def)));
}

static Args parse_args(int argc, char** argv) {
    Args a;

    a.cells        = get_arg_int(argc, argv, "--cells", a.cells);
    a.steps        = get_arg_int(argc, argv, "--steps", a.steps);
    a.sample_start = get_arg_int(argc, argv, "--sample-start", a.sample_start);
    a.sample_every = get_arg_int(argc, argv, "--sample-every", a.sample_every);
    a.out_every    = get_arg_int(argc, argv, "--out-every", a.out_every);
    a.seed         = get_arg_int(argc, argv, "--seed", a.seed);

    a.M1           = get_arg_double(argc, argv, "--M1", a.M1);
    a.gamma        = get_arg_double(argc, argv, "--gamma", a.gamma);
    a.L            = get_arg_double(argc, argv, "--L", a.L);
    a.dt           = get_arg_double(argc, argv, "--dt", a.dt);
    a.ppc          = get_arg_double(argc, argv, "--ppc", a.ppc);
    a.sigmaT       = get_arg_double(argc, argv, "--sigmaT", a.sigmaT);
    a.gmax_factor  = get_arg_double(argc, argv, "--gmax-factor", a.gmax_factor);

    a.prefix       = get_arg(argc, argv, "--prefix", a.prefix);

    return a;
}

static GasState upstream_state(const Args& a) {
    GasState s;
    s.n = 1.0;
    s.T = 1.0;
    s.u = a.M1 * std::sqrt(a.gamma * s.T);
    return s;
}

static GasState downstream_state(const Args& a) {
    GasState s1 = upstream_state(a);

    double g = a.gamma;
    double M = a.M1;

    double rho_ratio = ((g + 1.0)*M*M) / ((g - 1.0)*M*M + 2.0);
    double p_ratio   = 1.0 + (2.0*g/(g + 1.0))*(M*M - 1.0);

    GasState s2;
    s2.n = s1.n * rho_ratio;
    s2.T = s1.T * p_ratio / rho_ratio;
    s2.u = s1.u / rho_ratio;
    return s2;
}

static double normal_pdf(double a) {
    return std::exp(-0.5*a*a) / std::sqrt(2.0*PI);
}

static double normal_cdf(double a) {
    return 0.5*(1.0 + std::erf(a/std::sqrt(2.0)));
}

// Flux of Maxwellian molecules crossing a boundary into the domain.
// un is mean normal velocity into the domain.
static double maxwell_flux(double n, double un, double T) {
    double sig = std::sqrt(T);
    double a = un / sig;
    return n * (sig*normal_pdf(a) + un*normal_cdf(a));
}

// Sample v_n > 0 from the incoming flux distribution v_n f(v_n).
static double sample_flux_normal(double un, double T, std::mt19937_64& rng) {
    double sig = std::sqrt(T);
    double vmax = std::fabs(un) + 8.0*sig;
    if (vmax < 8.0*sig) vmax = 8.0*sig;

    std::normal_distribution<double> nd(un, sig);
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    while (true) {
        double v = nd(rng);
        if (v <= 0.0) continue;

        double p = v / vmax;
        if (p > 1.0) p = 1.0;

        if (ur(rng) < p) return v;
    }
}

static void random_unit_vector(double& ex, double& ey, double& ez, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    double mu = 2.0*ur(rng) - 1.0;
    double phi = 2.0*PI*ur(rng);
    double s = std::sqrt(std::max(0.0, 1.0 - mu*mu));

    ex = mu;
    ey = s*std::cos(phi);
    ez = s*std::sin(phi);
}

static int cell_index(double x, double xmin, double dx, int cells) {
    int c = (int)std::floor((x - xmin)/dx);
    if (c < 0) c = 0;
    if (c >= cells) c = cells - 1;
    return c;
}

static void build_bins(const Particles& p, vector<vector<int>>& bins, double xmin, double dx, int cells) {
    for (int c=0; c<cells; c++) bins[c].clear();

    for (int i=0; i<(int)p.size(); i++) {
        int c = cell_index(p.x[i], xmin, dx, cells);
        bins[c].push_back(i);
    }
}

static void compact_inside(Particles& p, double xmin, double xmax) {
    size_t w = 0;
    size_t n = p.size();

    for (size_t i=0; i<n; i++) {
        if (p.x[i] >= xmin && p.x[i] < xmax) {
            if (w != i) {
                p.x[w]  = p.x[i];
                p.vx[w] = p.vx[i];
                p.vy[w] = p.vy[i];
                p.vz[w] = p.vz[i];
            }
            w++;
        }
    }

    p.resize(w);
}

static void inject_boundary(
    Particles& p,
    const GasState& s,
    bool left_boundary,
    double xmin,
    double xmax,
    double dt,
    double FNUM,
    std::mt19937_64& rng
) {
    // Inward normal is +x at left boundary and -x at right boundary.
    double un = left_boundary ? s.u : -s.u;
    double flux = maxwell_flux(s.n, un, s.T);

    std::poisson_distribution<int> pois(flux * dt / FNUM);
    int Nin = pois(rng);

    std::normal_distribution<double> nd0(0.0, std::sqrt(s.T));
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    for (int k=0; k<Nin; k++) {
        double vn = sample_flux_normal(un, s.T, rng);
        double vx = left_boundary ? vn : -vn;
        double vy = nd0(rng);
        double vz = nd0(rng);

        // Inject uniformly over the part of the trajectory inside the time step.
        double tau = ur(rng)*dt;
        double xx = left_boundary ? xmin + vx*tau : xmax + vx*tau;

        if (xx >= xmin && xx < xmax) {
            p.push(xx, vx, vy, vz);
        }
    }
}

static void initialise_particles(
    Particles& p,
    const Args& a,
    const GasState& Ls,
    const GasState& Rs,
    double xmin,
    double xmax,
    double dx,
    double FNUM,
    std::mt19937_64& rng
) {
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    p.reserve((size_t)(a.cells*a.ppc*3.0));

    for (int c=0; c<a.cells; c++) {
        double xlo = xmin + c*dx;
        double xhi = xlo + dx;
        double xc = 0.5*(xlo + xhi);

        const GasState& s = (xc < 0.0) ? Ls : Rs;

        std::poisson_distribution<int> pois(s.n * dx / FNUM);
        int N = pois(rng);

        std::normal_distribution<double> ndx(s.u, std::sqrt(s.T));
        std::normal_distribution<double> nd0(0.0, std::sqrt(s.T));

        for (int i=0; i<N; i++) {
            double xx = xlo + ur(rng)*dx;
            p.push(xx, ndx(rng), nd0(rng), nd0(rng));
        }
    }
}

static void collide_ntc(
    Particles& p,
    vector<vector<int>>& bins,
    vector<double>& gmax,
    vector<double>& rem,
    ProfileAccum& acc,
    bool collect_prod,
    const Args& a,
    double xmin,
    double dx,
    double FNUM,
    std::mt19937_64& rng
) {
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    for (int c=0; c<a.cells; c++) {
        auto& ids = bins[c];
        int N = (int)ids.size();
        if (N < 2) continue;

        // Local mean velocity for central-moment production.
        double Ux=0.0, Uy=0.0, Uz=0.0;
        for (int id: ids) {
            Ux += p.vx[id];
            Uy += p.vy[id];
            Uz += p.vz[id];
        }
        Ux /= N;
        Uy /= N;
        Uz /= N;

        double Vcell = dx; // unit area

        double ncand_f =
            0.5 * N * (N - 1) * FNUM * a.sigmaT * gmax[c] * a.dt / Vcell
            + rem[c];

        int ncand = (int)std::floor(ncand_f);
        rem[c] = ncand_f - ncand;

        if (ncand <= 0) continue;

        std::uniform_int_distribution<int> pick(0, N-1);

        for (int m=0; m<ncand; m++) {
            int ia = pick(rng);
            int ib = pick(rng);
            if (ia == ib) continue;

            int i = ids[ia];
            int j = ids[ib];

            double dvx = p.vx[i] - p.vx[j];
            double dvy = p.vy[i] - p.vy[j];
            double dvz = p.vz[i] - p.vz[j];

            double g = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

            bool accept = false;
            if (g > gmax[c]) {
                gmax[c] = 1.2*g;
                accept = true;
            } else {
                accept = (ur(rng) < g/gmax[c]);
            }

            if (!accept) continue;

            // Pre-collision central fourth scalar moment for this collision pair.
            double c1x = p.vx[i] - Ux;
            double c1y = p.vy[i] - Uy;
            double c1z = p.vz[i] - Uz;

            double c2x = p.vx[j] - Ux;
            double c2y = p.vy[j] - Uy;
            double c2z = p.vz[j] - Uz;

            double c12 = c1x*c1x + c1y*c1y + c1z*c1z;
            double c22 = c2x*c2x + c2y*c2y + c2z*c2z;

            double c4_before = c12*c12 + c22*c22;

            // Elastic hard-sphere scattering with isotropic post-collision relative velocity.
            double Vx = 0.5*(p.vx[i] + p.vx[j]);
            double Vy = 0.5*(p.vy[i] + p.vy[j]);
            double Vz = 0.5*(p.vz[i] + p.vz[j]);

            double ex, ey, ez;
            random_unit_vector(ex, ey, ez, rng);

            double gvx = g*ex;
            double gvy = g*ey;
            double gvz = g*ez;

            p.vx[i] = Vx + 0.5*gvx;
            p.vy[i] = Vy + 0.5*gvy;
            p.vz[i] = Vz + 0.5*gvz;

            p.vx[j] = Vx - 0.5*gvx;
            p.vy[j] = Vy - 0.5*gvy;
            p.vz[j] = Vz - 0.5*gvz;

            if (collect_prod) {
                double d1x = p.vx[i] - Ux;
                double d1y = p.vy[i] - Uy;
                double d1z = p.vz[i] - Uz;

                double d2x = p.vx[j] - Ux;
                double d2y = p.vy[j] - Uy;
                double d2z = p.vz[j] - Uz;

                double d12 = d1x*d1x + d1y*d1y + d1z*d1z;
                double d22 = d2x*d2x + d2y*d2y + d2z*d2z;

                double c4_after = d12*d12 + d22*d22;

                // Production per unit volume accumulated over physical time.
                acc.prod_c4[c] += FNUM * (c4_after - c4_before) / Vcell;
                acc.prod_coll[c] += 1.0;
            }
        }
    }
}

static void sample_profile(
    const Particles& p,
    const vector<vector<int>>& bins,
    ProfileAccum& acc,
    double dx,
    double FNUM
) {
    acc.nsamp += 1;

    for (int c=0; c<acc.cells; c++) {
        const auto& ids = bins[c];
        int N = (int)ids.size();
        if (N < 1) continue;

        double n = FNUM * N / dx;

        double Ux=0.0, Uy=0.0, Uz=0.0;
        for (int id: ids) {
            Ux += p.vx[id];
            Uy += p.vy[id];
            Uz += p.vz[id];
        }
        Ux /= N;
        Uy /= N;
        Uz /= N;

        double sum_c2=0.0;
        double sum_q=0.0;
        double sum_sig=0.0;
        double sum_c4=0.0;
        double sum_Rkern=0.0;

        double sum_JDelta=0.0;
        double sum_KDelta=0.0;

        for (int id: ids) {
            double cx = p.vx[id] - Ux;
            double cy = p.vy[id] - Uy;
            double cz = p.vz[id] - Uz;

            double c2 = cx*cx + cy*cy + cz*cz;
            double c4 = c2*c2;

            sum_c2 += c2;
            sum_q += 0.5*c2*cx;
            sum_sig += cx*cx - c2/3.0;
            sum_c4 += c4;
            sum_Rkern += c2*(cx*cx - c2/3.0);

            // Scalar-excess budget flux terms.
            sum_JDelta += p.vx[id] * c4;
            sum_KDelta += p.vx[id] * c2 * cx;
        }

        double T = (sum_c2/N)/3.0;
        double qx = n * (sum_q/N);
        double sig = n * (sum_sig/N);
        double c4mom = n * (sum_c4/N);
        double Delta = c4mom - 15.0*n*T*T;
        double Rxx = n*(sum_Rkern/N) - 7.0*T*sig;
        double S = Rxx + Delta/3.0;

        double JDelta = n * (sum_JDelta/N);
        double KDelta = n * (sum_KDelta/N);

        acc.n[c] += n;
        acc.ux[c] += Ux;
        acc.uy[c] += Uy;
        acc.uz[c] += Uz;
        acc.T[c] += T;

        acc.qx[c] += qx;
        acc.sigxx[c] += sig;
        acc.c4[c] += c4mom;
        acc.Delta[c] += Delta;
        acc.Rxx[c] += Rxx;
        acc.S[c] += S;

        acc.JDelta[c] += JDelta;
        acc.KDelta[c] += KDelta;
    }
}

static void write_profiles(
    const string& filename,
    const ProfileAccum& acc,
    double xmin,
    double dx,
    double sample_time
) {
    ofstream out(filename);
    out << std::setprecision(16);

    out << "x,n,ux,uy,uz,T,qx,sigma_xx,c4_moment,Delta,Rxx_closure,S,"
        << "JDelta,KDelta,QDelta_collision,collision_count,nsamples,sample_time\n";

    double invs = (acc.nsamp > 0) ? 1.0/(double)acc.nsamp : 0.0;

    for (int c=0; c<acc.cells; c++) {
        double x = xmin + (c + 0.5)*dx;

        double n   = acc.n[c]*invs;
        double ux  = acc.ux[c]*invs;
        double uy  = acc.uy[c]*invs;
        double uz  = acc.uz[c]*invs;
        double T   = acc.T[c]*invs;

        double qx  = acc.qx[c]*invs;
        double sig = acc.sigxx[c]*invs;
        double c4  = acc.c4[c]*invs;
        double D   = acc.Delta[c]*invs;
        double R   = acc.Rxx[c]*invs;
        double S   = acc.S[c]*invs;

        double Jd  = acc.JDelta[c]*invs;
        double Kd  = acc.KDelta[c]*invs;

        double QD = (sample_time > 0.0) ? acc.prod_c4[c]/sample_time : 0.0;

        out << x << "," << n << "," << ux << "," << uy << "," << uz << ","
            << T << "," << qx << "," << sig << "," << c4 << ","
            << D << "," << R << "," << S << ","
            << Jd << "," << Kd << ","
            << QD << "," << acc.prod_coll[c] << ","
            << acc.nsamp << "," << sample_time << "\n";
    }

    out.close();
}


template<class T> void put(ostream& s, const T& v) { s.write((const char*)&v, sizeof(v)); }
template<class T> void get(istream& s, T& v) { s.read((char*)&v, sizeof(v)); }
void putvec(ostream& s, const vector<double>& v) {
    size_t n=v.size(); put(s,n); s.write((const char*)v.data(), n*sizeof(double));
}
void getvec(istream& s, vector<double>& v) {
    size_t n=0; get(s,n); if(n>100000000) throw runtime_error("Invalid checkpoint vector");
    v.resize(n); s.read((char*)v.data(), n*sizeof(double));
}
string signature(const Args& a) {
    ostringstream s; s<<setprecision(17)<<a.cells<<' '<<a.sample_start<<' '<<a.sample_every
    <<' '<<a.seed<<' '<<a.M1<<' '<<a.gamma<<' '<<a.L<<' '<<a.dt<<' '<<a.ppc
    <<' '<<a.sigmaT<<' '<<a.gmax_factor; return s.str();
}
void save_ck(const Args& a, int step, const Particles& p, const vector<double>& gmax,
 const vector<double>& rem, const ProfileAccum& acc, double time, double st, bool sampling,
 const mt19937_64& rng) {
    string tmp=a.prefix+".checkpoint.tmp"; ofstream o(tmp,ios::binary);
    o<<"JFM1439_C4_V1\n"<<signature(a)<<'\n'<<rng<<'\n';
    put(o,step); put(o,time); put(o,st); put(o,sampling); put(o,acc.nsamp);
    putvec(o,p.x); putvec(o,p.vx); putvec(o,p.vy); putvec(o,p.vz);
    putvec(o,gmax); putvec(o,rem);
    putvec(o,acc.n);
    putvec(o,acc.ux);
    putvec(o,acc.uy);
    putvec(o,acc.uz);
    putvec(o,acc.T);
    putvec(o,acc.qx);
    putvec(o,acc.sigxx);
    putvec(o,acc.c4);
    putvec(o,acc.Delta);
    putvec(o,acc.Rxx);
    putvec(o,acc.S);
    putvec(o,acc.JDelta);
    putvec(o,acc.KDelta);
    putvec(o,acc.prod_c4);
    putvec(o,acc.prod_coll);
    o.flush(); if(!o) throw runtime_error("Checkpoint write failed"); o.close();
    if(rename(tmp.c_str(),(a.prefix+".checkpoint").c_str())) throw runtime_error("Checkpoint rename failed");
}
int load_ck(const Args& a, Particles& p, vector<double>& gmax, vector<double>& rem,
 ProfileAccum& acc, double& time, double& st, bool& sampling, mt19937_64& rng) {
    ifstream i(a.prefix+".checkpoint",ios::binary); if(!i) return 0;
    string magic,sig; getline(i,magic); getline(i,sig);
    if(magic!="JFM1439_C4_V1" || sig!=signature(a)) throw runtime_error("Checkpoint mismatch");
    i>>rng; i.get(); int step=0;
    get(i,step); get(i,time); get(i,st); get(i,sampling); get(i,acc.nsamp);
    getvec(i,p.x); getvec(i,p.vx); getvec(i,p.vy); getvec(i,p.vz);
    getvec(i,gmax); getvec(i,rem);
    getvec(i,acc.n);
    getvec(i,acc.ux);
    getvec(i,acc.uy);
    getvec(i,acc.uz);
    getvec(i,acc.T);
    getvec(i,acc.qx);
    getvec(i,acc.sigxx);
    getvec(i,acc.c4);
    getvec(i,acc.Delta);
    getvec(i,acc.Rxx);
    getvec(i,acc.S);
    getvec(i,acc.JDelta);
    getvec(i,acc.KDelta);
    getvec(i,acc.prod_c4);
    getvec(i,acc.prod_coll);
    if(!i || step>a.steps || p.x.size()!=p.vx.size() || gmax.size()!=size_t(a.cells))
        throw runtime_error("Invalid checkpoint");
    cout<<"[restart] step="<<step<<endl; return step;
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    GasState Ls = upstream_state(a);
    GasState Rs = downstream_state(a);

    double xmin = -0.5*a.L;
    double xmax =  0.5*a.L;
    double dx = a.L / a.cells;

    // Upstream ppc fixes particle weight. Unit area.
    double FNUM = Ls.n * dx / a.ppc;

    std::mt19937_64 rng(a.seed);

    Particles p;
    initialise_particles(p, a, Ls, Rs, xmin, xmax, dx, FNUM, rng);

    vector<vector<int>> bins(a.cells);

    double g0 = 2.0*std::fabs(Ls.u - Rs.u)
              + a.gmax_factor*std::sqrt(std::max(Ls.T, Rs.T));

    vector<double> gmax(a.cells, g0);
    vector<double> rem(a.cells, 0.0);

    ProfileAccum acc(a.cells);

    double time = 0.0;
    double sample_time = 0.0;
    bool sampling_started = false;

    cout << "============================================================\n";
    cout << "Standalone 1D3V DSMC normal shock with c4 production\n";
    cout << "M1=" << a.M1 << " gamma=" << a.gamma << "\n";
    cout << "Left : n=" << Ls.n << " u=" << Ls.u << " T=" << Ls.T << "\n";
    cout << "Right: n=" << Rs.n << " u=" << Rs.u << " T=" << Rs.T << "\n";
    cout << "cells=" << a.cells << " L=" << a.L << " dx=" << dx << "\n";
    cout << "dt=" << a.dt << " steps=" << a.steps << " sample_start=" << a.sample_start << "\n";
    cout << "ppc_upstream=" << a.ppc << " FNUM=" << FNUM << "\n";
    cout << "initial particles=" << p.size() << "\n";
    cout << "sigmaT=" << a.sigmaT << "\n";
    cout << "prefix=" << a.prefix << "\n";
    cout << "============================================================\n";

    int start_step=load_ck(a,p,gmax,rem,acc,time,sample_time,sampling_started,rng);
    for (int step=start_step+1; step<=a.steps; step++) {
        time += a.dt;

        // Move particles.
        for (size_t i=0; i<p.size(); i++) {
            p.x[i] += p.vx[i]*a.dt;
        }

        // Remove escaped particles.
        compact_inside(p, xmin, xmax);

        // Reservoir injection.
        inject_boundary(p, Ls, true, xmin, xmax, a.dt, FNUM, rng);
        inject_boundary(p, Rs, false, xmin, xmax, a.dt, FNUM, rng);

        // Sort into cells.
        build_bins(p, bins, xmin, dx, a.cells);

        if (step == a.sample_start) {
            acc.reset(a.cells);
            sample_time = 0.0;
            sampling_started = true;
        }

        bool collect_prod = sampling_started;

        collide_ntc(p, bins, gmax, rem, acc, collect_prod, a, xmin, dx, FNUM, rng);

        if (sampling_started) sample_time += a.dt;

        // Sample after collisions.
        if (sampling_started && ((step - a.sample_start) % a.sample_every == 0)) {
            build_bins(p, bins, xmin, dx, a.cells);
            sample_profile(p, bins, acc, dx, FNUM);
        }

        if (step % 1000 == 0 || step == a.steps)
            save_ck(a,step,p,gmax,rem,acc,time,sample_time,sampling_started,rng);
        if (step % a.out_every == 0 || step == a.steps) {
            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s_step%08d.csv", a.prefix.c_str(), step);
            write_profiles(fname, acc, xmin, dx, sample_time);

            cout << "[step " << step << "]"
                 << " time=" << time
                 << " Np=" << p.size()
                 << " nsamp=" << acc.nsamp
                 << " sample_time=" << sample_time
                 << " wrote=" << fname << "\n";
        }
    }

    string finalname = a.prefix + "_final.csv";
    write_profiles(finalname, acc, xmin, dx, sample_time);
    cout << "[done] wrote " << finalname << "\n";

    return 0;
}
