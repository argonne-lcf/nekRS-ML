#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

nek_data = sys.argv[1]
Retau    = float(sys.argv[2])
tw       = float(sys.argv[3])

strIdx = 100;


decimals=5

plt.close("all");
plt.style.use('classic');
font = {'family' : 'DejaVu Serif',
        'weight' : 'normal',
        'size'   : 24};
plt.rc('font', **font);

mpl.rcParams['xtick.major.size'] = 5;
mpl.rcParams['xtick.major.width'] = 2;
mpl.rcParams['xtick.minor.size'] = 2.5;
mpl.rcParams['xtick.minor.width'] = 2;

mpl.rcParams['ytick.major.size'] = 5;
mpl.rcParams['ytick.major.width'] = 2;
mpl.rcParams['ytick.minor.size'] = 2.5;
mpl.rcParams['ytick.minor.width'] = 2;

#inputs
indices=['Points:0','AVVEL:0','AVPRE','AVVE2:0','AVVXY:0','AVRHO','AVMUE','AVVGR:0','AVVTW:1'];
index_var = np.zeros(len(indices),dtype=int);

fid = open(nek_data); 
var = fid.readline();
var = var.replace('"','')
var = var.split(",")
for i in range(len(indices)):
  try:
    ind = var.index(indices[i])
  except:
    ind = -1
  index_var[i] = ind;

iX  = index_var[0];
iU  = index_var[1];
iP  = index_var[2];
iUU = index_var[3];
iUV = index_var[4];
iR  = index_var[5];
iMU = index_var[6];
idV = index_var[7];
idT = index_var[8];

#inputs
les = np.loadtxt(nek_data, delimiter=',', skiprows=1);
fname = 'Re{}_DNS.txt'.format(int(Retau))
dns = np.loadtxt(fname, delimiter=','); 

Uref=1.0;
delta =1.0;
rho = 1;
Re = np.exp((1.0/0.88)*np.log(Retau/0.09));
mu = (rho*2.0*delta*Uref)/Re; 
utau = (Retau*mu)/(delta*rho); 
print('--||SOD :: Input mu ',mu);
print('--||SOD :: Theory Reb ',Re);
print('--||SOD :: Theory utau ',utau);
print('--||SOD :: Theory PG  ',utau*utau*rho/delta);

# projection to cartesian mesh

x   = les[:,iX];
y   = les[:,iX+1];
if(np.amin(y)<0):
  y = y+1.0
z   = les[:,iX+2];
bl_u   = les[:,iU];
bl_v   = les[:,iU+1];
bl_w   = les[:,iU+2];
bl_r   = les[:,iR];
if(index_var[5]<0):
  bl_r = 0.0*bl_r+1.0;
bl_p   = les[:,iP];
bl_uu  = les[:,iUU]-np.square(les[:,iU]);
bl_vv  = les[:,iUU+1];
bl_ww  = les[:,iUU+2];
bl_uv  = les[:,iUV];
bl_uw  = les[:,iUV+1];
bl_vw  = les[:,iUV+2]
bl_mue = les[:,iMU];
bl_dudx = les[:,idV];
bl_dudy = les[:,idV+1];
bl_dudz = les[:,idV+2];
bl_dvdx = les[:,idV+3];
bl_dvdy = les[:,idV+4];
bl_dvdz = les[:,idV+5];
bl_dwdx = les[:,idV+6];
bl_dwdy = les[:,idV+7];
bl_dwdz = les[:,idV+8];
bl_avtw = les[:,idT];


#xlin = np.unique(np.around(x,decimals));
#ylin = np.unique(np.around(y,decimals));
#zlin = np.unique(np.around(z,decimals));

xlin = x;
ylin = y;
zlin = z;

indSort = np.lexsort((z,x,y))

print('--||SOD :: XDOMAIN = %.2f -- %.2f' % (np.amin(xlin),np.amax(xlin)))
print('--||SOD :: YDOMAIN = %.2f -- %.2f' % (np.amin(ylin),np.amax(ylin)))
print('--||SOD :: ZDOMAIN = %.2f -- %.2f' % (np.amin(zlin),np.amax(zlin)))

nx = len(xlin)
ny = len(ylin)
nz = len(zlin)

print('--||SOD :: Mesh size : nx[{}] ny[{}] nz[{}]'.format(len(xlin),len(ylin),len(zlin)))

midline_size = int(0.5*ny);
line_size    = ny;
delta_nrs = (np.amax(ylin) - np.amin(ylin)) / 2

print('--||SOD :: mid_line',midline_size )

avgmu = mu;

if(tw==0):
  tw = utau*utau

utauDNS  = utau
utau = np.sqrt(tw/rho);
ubulk = np.trapz(bl_u,ylin)/(np.amax(ylin)-np.amin(ylin))

Re_sim = ((np.amax(ylin)-np.amin(ylin))*ubulk/avgmu);
Re_tau_sim = utau/avgmu;
#Re_tau_sim = 0.09*np.exp(0.88*np.log(Re_sim));
#utauDNS = (Re_tau_sim*avgmu)
ubulkDNS = np.trapz(utauDNS*dns[:,2],dns[:,0])/(np.amax(dns[:,0])-np.amin(dns[:,0]))

err_utau = 100*(utau/ubulk-utauDNS/ubulkDNS)/(utauDNS/ubulkDNS)

print('--||SOD :: DNS Data Ub ',ubulkDNS)
print('--||SOD :: DNS Data utau ',utauDNS/ubulkDNS)
print('--||SOD :: Simulation mu ',avgmu)
print('--||SOD :: Simulation Ub ',ubulk)
print('--||SOD :: Simulation Reb ',Re_sim)
print('--||SOD :: Simulation utau ',utau/ubulk)
print('--||SOD :: Simulation Retau ',Re_tau_sim)
print('--||SOD :: Error tw %',err_utau)

bl_ustar   = np.zeros(midline_size);
bl_ystar   = np.zeros(midline_size);
bl_uustar  = np.zeros(midline_size);
bl_vvstar  = np.zeros(midline_size);
bl_wwstar  = np.zeros(midline_size);
bl_uvstar  = np.zeros(midline_size);
bl_uwstar  = np.zeros(midline_size);
bl_vwstar  = np.zeros(midline_size);

for j in range(midline_size):
    bl_ystar[j]  = ylin[j]*utau/avgmu;
    #bl_ystar[j]  = ylin[j]*utau/mu;
    bl_ustar[j]  = bl_u[j]/utau;
    bl_uustar[j] = np.sqrt(np.abs(bl_uu[j]))/(utau);
    bl_vvstar[j] = np.sqrt(np.abs(bl_vv[j]))/(utau);
    bl_wwstar[j] = np.sqrt(np.abs(bl_ww[j]))/(utau);
    bl_uvstar[j] = np.sqrt(np.abs(bl_uv[j]))/(utau);
    bl_uwstar[j] = np.sqrt(np.abs(bl_uw[j]))/(utau);
    bl_vwstar[j] = np.sqrt(np.abs(bl_vw[j]))/(utau);

print('--||SOD :: Bl averages done.')
print('--||SOD :: dy+ :', np.amin(np.diff(bl_ystar)), np.amax(np.diff(bl_ystar)))

# print the data

# bl plots
lbl = r'NekRS: ${} \times {} \times {}$'.format(len(ylin),len(ylin),len(ylin))
dns_lbl = r'DNS: $(768 \times 768 \times 385)$'

# U + 
fig=plt.figure(1)

plt.plot(bl_ystar[1:],bl_ustar[1:],'b',linewidth=3.0,label=lbl)
plt.plot(dns[:,1],dns[:,2],'k--',linewidth=3.0,label=dns_lbl)
plt.axis([0.1, 2*Retau, 0, 1.2*np.amax(bl_ustar[1:],axis=None) ])
plt.xscale('log')
plt.ylabel(r'$U^+$')
plt.xlabel(r'$y^+$')
plt.tight_layout()
plt.grid()
legend = plt.legend(loc='upper left', fontsize=11)
plt.savefig('U+.png')


# Urms + 
fig=plt.figure(2)

plt.plot(bl_ystar[1:],bl_uustar[1:],'b',linewidth=3.0,label=r'$uu^+$')
plt.plot(bl_ystar[1:],bl_vvstar[1:],'r',linewidth=3.0,label=r'$vv^+$')
plt.plot(bl_ystar[1:],bl_wwstar[1:],'g',linewidth=3.0,label=r'$ww^+$')
plt.plot(dns[1:,1],dns[1:,3],'k--',linewidth=3.0,label=dns_lbl)
plt.plot(dns[:,1],dns[:,4],'k--',linewidth=3.0)
plt.plot(dns[:,1],dns[:,5],'k--',linewidth=3.0)
plt.axis([0.1, 2*Retau, 0, 1.2*np.amax(bl_uustar[1:],axis=None)])
plt.xscale('log')
plt.ylabel(r'$U^{\prime +}$')
plt.xlabel(r'$y^+$')
plt.tight_layout()
plt.grid()
plt.savefig('Urms+.png')

# Urms + 
fig=plt.figure(3)

plt.plot(ylin[:midline_size]/delta_nrs,bl_uu[:midline_size],'b',linewidth=3.0,label=r'$uu$')
plt.plot(ylin[:midline_size]/delta_nrs,bl_vv[:midline_size],'r',linewidth=3.0,label=r'$vv$')
plt.plot(ylin[:midline_size]/delta_nrs,bl_ww[:midline_size],'g',linewidth=3.0,label=r'$ww$')
xdata = dns[:,0]
ydata = np.square(utauDNS*dns[:,3]/ubulkDNS)
plt.plot(xdata,ydata,'k--',linewidth=3.0,label=dns_lbl)
xdata = dns[:,0]
ydata = np.square(utauDNS*dns[:,4]/ubulkDNS)
plt.plot(xdata,ydata,'k--',linewidth=3.0)
xdata = dns[:,0]
ydata = np.square(utauDNS*dns[:,5]/ubulkDNS)
plt.plot(xdata,ydata,'k--',linewidth=3.0)
plt.ylabel(r'$U^{\prime}U^{\prime}$')
plt.xlabel(r'$y/\delta$')
plt.tight_layout()
plt.grid()
plt.savefig('Urms.png')


# UVrms + 
fig=plt.figure(4)

plt.plot(bl_ystar[1:],bl_uvstar[1:],'b',linewidth=3.0,label=lbl)
plt.plot(dns[:,1],np.sqrt(-dns[:,10]),'k--',linewidth=3.0,label=dns_lbl)
plt.axis([0.1, 2*Retau, 0, 1.2*np.amax(bl_uvstar[1:],axis=None)])
plt.xscale('log')
plt.ylabel(r'$U^{\prime}V^{\prime +}$')
plt.xlabel(r'$y^+$')
plt.tight_layout()
plt.grid()
plt.savefig('UVrms+.png')

# UVrms
fig=plt.figure(5)

plt.plot(ylin[:midline_size]/delta_nrs,-bl_uv[:midline_size],'b',linewidth=3.0,label=lbl)
xdata = dns[:,0]
ydata = -utauDNS**2*dns[:,10]/ubulkDNS**2
plt.plot(xdata,ydata,'k--',linewidth=3.0)
#plt.plot(dns[:,1],-dns[:,10],'k--',linewidth=3.0,label=dns_lbl)
plt.ylabel(r'$U^{\prime}V^{\prime}$')
plt.xlabel(r'$y/\delta$')
plt.tight_layout()
plt.grid()
plt.ylim(0,0.0025)
plt.savefig('UVrms.png')

# U  
xdata = dns[:,0]
ydata = dns[:,2]
ydata = utauDNS*ydata/ubulkDNS

fig=plt.figure(6)

plt.plot(ylin[:midline_size]/delta_nrs,bl_u[:midline_size]/ubulk,'b',linewidth=3.0,label=lbl)
plt.plot(xdata,ydata,'k--',linewidth=3.0,label=dns_lbl)
plt.ylabel(r'$U/U_b$')
plt.xlabel(r'$y/\delta$')
plt.tight_layout()
plt.grid()
plt.savefig('U.png')
