#ifndef GROPT_BINDING_H
#define GROPT_BINDING_H

#ifdef __cplusplus
extern "C" {
#endif

void run_kernel_diff_fixeddt(double **G_out, int *N_out, double **ddebug,
                             int verbose, double dt0, double gmax, double smax, double TE,
                             int N_moments, const double *moments_params, double PNS_thresh,
                             double T_readout, double T_90, double T_180, int diffmode,
                             double dt_out, int N_eddy, const double *eddy_params,
                             double search_bval, double slew_reg, int Naxis);

void run_kernel_diff_fixedN(double **G_out, int *N_out, double **ddebug,
                            int verbose, int N0, double gmax, double smax, double TE,
                            int N_moments, const double *moments_params, double PNS_thresh,
                            double T_readout, double T_90, double T_180, int diffmode,
                            double dt_out, int N_eddy, const double *eddy_params,
                            double search_bval, double slew_reg);

void run_kernel_diff_fixedN_Gin(double **G_out, int *N_out, double **ddebug,
                                int verbose, const double *G_in, int N0, double gmax, double smax, double TE,
                                int N_moments, const double *moments_params, double PNS_thresh,
                                double T_readout, double T_90, double T_180, int diffmode,
                                double dt_out, int N_eddy, const double *eddy_params,
                                double search_bval, double slew_reg);

void run_kernel_diff_fixeddt_fixG(double **G_out, int *N_out, double **ddebug,
                                  int verbose, double dt0, double gmax, double smax, double TE,
                                  int N_moments, const double *moments_params, double PNS_thresh,
                                  double T_readout, double T_90, double T_180, int diffmode,
                                  double dt_out, int N_eddy, const double *eddy_params,
                                  double search_bval, int N_gfix, const double *gfix,
                                  double slew_reg);

#ifdef __cplusplus
}
#endif

#endif // GROPT_BINDING_H