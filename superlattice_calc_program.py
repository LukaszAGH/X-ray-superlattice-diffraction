import math
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
import cProfile
import psutil
import pstats
import io
from memory_profiler import profile, memory_usage
from concurrent.futures import ThreadPoolExecutor

class LCG:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        
    def next(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed / self.m

def box_muller(lcg):
    u1 = random.random()
    u2 = random.random()
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
    return z0, z1
    
def plot_xray_spectra_3(minTheta, maxTheta):

    data_factor = np.loadtxt('res_fac_06_th_' + str(minTheta) + '_' + str(maxTheta) + '.txt')
    data_basic = np.loadtxt('res_bas_06_th_' + str(minTheta) + '_' + str(maxTheta) + '.txt')
    data_bl = np.loadtxt('res_bl_06_th_' + str(minTheta) + '_' + str(maxTheta) + '.txt')
    #data_s = np.loadtxt('res_s_06_th_' + str(minTheta) + '_' + str(maxTheta) + '.txt')
    
    plt.figure(figsize=(12, 6))
    #plt.title("X-ray spectra", fontname="Arial", fontsize=16)
    plt.xlabel("2θ [degrees]", fontname="Arial", fontsize=16, fontweight='bold')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold')
    plt.ylabel("Intensity", fontname="Arial", fontsize=16, fontweight='bold')
    
    #range
    plt.xlim(minTheta, maxTheta)
    plt.ylim(auto=True)
    
    plt.grid(True)
    plt.yscale('log')
    
    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    
    #ticks
    plt.tick_params(axis='both', which='both', direction='out', width=1, length=5, pad=5)
    
    # Set line styles
    plt.plot(data_factor[:,0], data_factor[:,1], linestyle='-', linewidth=1.6, label="Superlattice factor")
    plt.plot(data_basic[:,0], data_basic[:,1], linestyle='-', linewidth=1.6, label="Superlattice basic")
    plt.plot(data_bl[:,0], data_bl[:,1], linestyle='-', linewidth=1.6, label="Superlattice bl")
    #plt.plot(data_s[:,0], data_s[:,1], linestyle='-', linewidth=1.6, label="S()")
    #legend
    #plt.legend(loc='lower left', fontsize=8)
    
    if minTheta == 40 and maxTheta == 50:
        # Add annotations for 40-50
        #plt.annotate('(d)', xy=(46.6, 6), xytext=(46.6, 155), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(43.2, 2.8), xytext=(43.5, 15), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(42.4, 19992), xytext=(42, 169604), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(a)', xy=(44.9, 9.4e6), xytext=(44.5, 6.6e7), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
    
    plt.savefig('xray_spectra_3_plot_' + str(minTheta) + '_' + str(maxTheta) + '.png', dpi=500, bbox_inches='tight')

    plt.show()

def plot_xray_spectra_pto(minTheta, maxTheta):

    # data_mc = np.loadtxt('res_pto_06_th_' + str(minTheta) + '_m_' + str(maxTheta) + '.txt')
    # data_w = np.loadtxt('res_pto_06_th_' + str(minTheta) + '_w_' + str(maxTheta) + '.txt')
    # data_basic = np.loadtxt('res_pto_06_th_' + str(minTheta) + '_' + str(maxTheta) + '.txt')
    
    data_mc = np.loadtxt('res_pto_06_th_' + str(minTheta) + '_m__' + str(maxTheta) + '.txt')
    data_w = np.loadtxt('res_pto_06_th_' + str(minTheta) + '_w__' + str(maxTheta) + '.txt')
    data_basic = np.loadtxt('res_pto_06_th_' + str(minTheta) + '__' + str(maxTheta) + '.txt')
    
    plt.figure(figsize=(12, 6))
    #plt.title("X-ray spectra", fontname="Arial", fontsize=16)
    plt.xlabel("2θ [degrees]", fontname="Arial", fontsize=16, fontweight='bold')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold')
    plt.ylabel("Intensity [arb. units]", fontname="Arial", fontsize=16, fontweight='bold')
    
    #range
    plt.xlim(minTheta, maxTheta)
    
    plt.grid(True)
    plt.yscale('log')
    
    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    
    #ticks
    plt.tick_params(axis='both', which='both', direction='out', width=1, length=5, pad=5)
    
    #Set line styles
    plt.plot(data_basic[:,0], data_basic[:,1] * 0.00001, linestyle='-', linewidth=1.6, label="PTO 1.5406", color='#d62728')
    plt.plot(data_w[:,0], data_w[:,1] * 0.00000001, linestyle='-', linewidth=1.6, label="PTO 1.5406 & 1.5444", color='#9467bd')
    plt.plot(data_mc[:,0], data_mc[:,1] * 0.00000000001, linestyle='-', linewidth=1.6, label="PTO 0.6", color='#e377c2')
    #legend
    #plt.legend(loc='upper right', fontsize=8)
    
    if minTheta == 10 and maxTheta == 50:
        # Add annotations for 10-50
        plt.annotate('(a)', xy=(24, 0.0002), xytext=(21, 1.59e-5), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(18, 3), xytext=(15.5, 0.55), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(21.7, 1828), xytext=(19, 22000), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        
    if minTheta == 10 and maxTheta == 60:
        # Add annotations for 10-60
        plt.annotate('(a)', xy=(22, 28.4), xytext=(20.5, 177), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(13.8, 2.9e-5), xytext=(12.7, 0.0002), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(24.2, 1.3e-7), xytext=(23.5, 1e-8), arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
    
    plt.savefig('xray_spectra_pto1_plot_' + str(minTheta) + '_' + str(maxTheta) + '.png', dpi=500, bbox_inches='tight')

    plt.show()

def plot_xray_spectra(minTheta, maxTheta, peaks=False, data_config='30-80', 
                     yscale='log', ylim=None, scaling_factors=None, 
                     colors=None, output_filename=None):
    """
    Plot X-ray spectra with flexible configuration while keeping fixed annotations.
    
    Parameters:
    - minTheta, maxTheta: angle range for the plot
    - peaks: whether to show peak annotations (only for 60-80 range)
    - data_config: which data files to load ('10-120', '30-80', or '60-80')
    - yscale: scale for y-axis ('log' or 'linear')
    - ylim: tuple for y-axis limits (min, max)
    - scaling_factors: dictionary with scaling factors for each dataset
    - colors: dictionary with colors for each dataset
    - output_filename: custom output filename (None for auto-generated)
    """
    
    if scaling_factors is None:
        if data_config == '60-80':
            scaling_factors = {'exp': 1, 'stressed': 1e6, 'bulk': 10}
        else:
            scaling_factors = {'exp': 1, 'stressed': 1e5, 'bulk': 1e9}
    
    if colors is None:
        colors = {
            'exp': '#d62728',
            'stressed': '#9467bd',
            'bulk': '#e377c2'
        }
    
    if data_config == '10-120':
        data_exp = np.loadtxt('sample_line_det.txt')
        data_stressed = np.loadtxt(f'res_lsmo_06_th_{minTheta}_{maxTheta}.txt')
        data_bulk = np.loadtxt(f'res_lsmo_06_th_{minTheta}_b_{maxTheta}.txt')
    elif data_config == '30-80':
        data_exp = np.loadtxt(f'res_lsmo_06_th_{minTheta}_1_{maxTheta}.txt')
        data_stressed = np.loadtxt(f'res_lsmo_06_th_{minTheta}_2_{maxTheta}.txt')
        data_bulk = np.loadtxt(f'res_lsmo_06_th_{minTheta}_3_{maxTheta}.txt')
    elif data_config == '60-80':
        data_exp = np.loadtxt('sample_line_det.txt')
        data_stressed = np.loadtxt(f'res_lsmo_06_th_{minTheta}_80_{maxTheta}.txt')
        data_bulk = np.loadtxt(f'res_lsmo_06_th_{minTheta}_60_{maxTheta}.txt')
    else:
        raise ValueError(f"Unknown data configuration: {data_config}")
    
    plt.figure(figsize=(12, 6))
    plt.xlabel("2θ [degrees]", fontname="Arial", fontsize=16, fontweight='bold')
    plt.ylabel("Intensity [arb. units]", fontname="Arial", fontsize=16, fontweight='bold')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold')
    
    plt.xlim(minTheta, maxTheta)
    if ylim:
        plt.ylim(*ylim)
    else:
        if data_config == '60-80':
            plt.ylim(0.00001, 10000000)
        elif data_config == '10-120':
            plt.ylim(0.000001, 10000000000)
        else:  # 30-80
            plt.ylim(0.0000001, 10000000000)
    
    plt.grid(True)
    plt.yscale(yscale)
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1)
    plt.tick_params(axis='both', which='both', direction='out', width=1, length=5, pad=5)
    
    plt.plot(data_exp[:,0], data_exp[:,1], linestyle='-', linewidth=1.6, 
             label="Experimental data", color=colors['exp'])
    plt.plot(data_stressed[:,0], data_stressed[:,1]/scaling_factors['stressed'], 
             linestyle='-', linewidth=1.6, label="Stressed structure spectra", 
             color=colors['stressed'])
    plt.plot(data_bulk[:,0], data_bulk[:,1]/scaling_factors['bulk'], 
             linestyle='-', linewidth=1.6, label="Bulk spectra", 
             color=colors['bulk'])
    
    if minTheta == 10 and maxTheta == 120:
        plt.annotate('(a)', xy=(46, 6e6), xytext=(39, 3.3e7), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(41.6, 10.9), xytext=(35.3, 102), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(42.3, 0.0005), xytext=(35.8, 0.005), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
    
    if minTheta == 40 and maxTheta == 50:
        plt.annotate('(a)', xy=(46.4, 4e6), xytext=(45.3, 6e6), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(42.6, 240), xytext=(41.5, 350), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(43.8, 0.74), xytext=(42.7, 1.5), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        
    if minTheta == 30 and maxTheta == 80:
        plt.annotate('(a)', xy=(44.7, 2.3e6), xytext=(43.3, 1e8), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(41.7, 0.7), xytext=(40.5, 30), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(42.7, 0.0003), xytext=(41.7, 0.006), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        
    if minTheta == 60 and maxTheta == 80:
        plt.annotate('(a)', xy=(72.5, 143134), xytext=(71.4, 1.73e6), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(b)', xy=(66.6, 10.6), xytext=(65.9, 63.9), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)
        plt.annotate('(c)', xy=(68.2, 0.01), xytext=(67.6, 0.08), 
                    arrowprops=dict(facecolor='black', arrowstyle='simple'), fontsize=14)

        if peaks:
            default_green = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
            default_orange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
            plt.vlines(x=70, ymin=0.0001, ymax=0.01, color=default_green, linewidth=3)
            plt.text(69.3, 0.00005, f'elong.   bare', color='black', rotation=0, 
                    ha='center', va='top', fontsize=10, fontweight='bold')
            plt.vlines(x=68.8, ymin=0.0001, ymax=0.01, color=default_orange, linewidth=3)
            plt.text(68, 0.0001, r'$(003)_{\mathrm{BTO}}$', color='black', rotation=0, 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            plt.vlines(x=73.4, ymin=0.0001, ymax=0.01, color=default_green, linewidth=3)
            plt.text(73.7, 0.00005, f'bare compr.', color='black', rotation=0, 
                    ha='center', va='top', fontsize=10, fontweight='bold')
            plt.vlines(x=73.7, ymin=0.0001, ymax=0.01, color=default_orange, linewidth=3)
            plt.text(74.6, 0.0001, r'$(003)_{\mathrm{LSMO}}$', color='black', rotation=0, 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if output_filename is None:
        output_filename = f'xray_spectra_plot_{minTheta}_{maxTheta}.png'
    plt.savefig(output_filename, dpi=500, bbox_inches='tight')
    plt.show()

def atsc(nhlp, S):
    a94 = [[0.0] * 200 for _ in range(4)]
    b94 = [[0.0] * 200 for _ in range(4)]
    c94 = [0.0] * 200
    
    a94[0][nhlp] = 0.0
    b94[0][nhlp] = 1.0
    a94[1][nhlp] = 0.0
    b94[1][nhlp] = 1.0
    a94[2][nhlp] = 0.0
    b94[2][nhlp] = 1.0
    a94[3][nhlp] = 0.0
    b94[3][nhlp] = 1.0
    c94[nhlp] = 0.0
    
    a94[0][8] = 3.5531
    b94[0][8] = 6.8702
    a94[1][8] = 2.6162
    b94[1][8] = 21.0743
    a94[2][8] = 1.2120
    b94[2][8] = 0.3871
    a94[3][8] = 0.6107
    b94[3][8] = 0.0960
    c94[8] = 0.0

    a94[0][13] = 3.4553
    b94[0][13] = 54.4948
    a94[1][13] = 7.1181
    b94[1][13] = 2.8378
    a94[2][13] = 1.0330
    b94[2][13] = 0.5062
    a94[3][13] = 1.3472
    b94[3][13] = 0.0621
    c94[13] = 0.0

    a94[0][22] = 4.7939
    b94[0][22] = 25.4898
    a94[1][22] = 7.2348
    b94[1][22] = 7.3826
    a94[2][22] = 7.8591
    b94[2][22] = 0.5846
    a94[3][22] = 1.6990
    b94[3][22] = 0.0293
    c94[22] = 0.0

    a94[0][25] = 14.2140
    b94[0][25] = 8.5904
    a94[1][25] = 0.7519
    b94[1][25] = 1.6230
    a94[2][25] = 7.3544
    b94[2][25] = 0.4327
    a94[3][25] = 1.7816
    b94[3][25] = 0.0250
    c94[25] = 0.0

    a94[0][38] = 10.8557
    b94[0][38] = 20.8810
    a94[1][38] = 18.4417
    b94[1][38] = 1.6734
    a94[2][38] = 6.6905
    b94[2][38] = 0.1318
    a94[3][38] = 1.4186
    b94[3][38] = 0.0054
    c94[38] = 0.0

    a94[0][44] = 17.4492
    b94[0][44] = 13.7038
    a94[1][44] = 16.1683
    b94[1][44] = 1.1046
    a94[2][44] = 4.7256
    b94[2][44] = 0.2453
    a94[3][44] = 5.0909
    b94[3][44] = 0.0380
    c94[44] = 0.0

    a94[0][56] = 10.6045
    b94[0][56] = 34.1617
    a94[1][56] = 21.2072
    b94[1][56] = 3.9127
    a94[2][56] = 17.7637
    b94[2][56] = 0.3884
    a94[3][56] = 6.0052
    b94[3][56] = 0.0228
    c94[56] = 0.0

    a94[0][57] = 11.3144
    b94[0][57] = 33.9508
    a94[1][57] = 21.4338
    b94[1][57] = 3.7166
    a94[2][57] = 17.5797
    b94[2][57] = 0.3777
    a94[3][57] = 6.2639
    b94[3][57] = 0.0252
    c94[57] = 0.0

    a94[0][82] = 23.6202
    b94[0][82] = 11.8888
    a94[1][82] = 35.6786
    b94[1][82] = 1.2275
    a94[2][82] = 17.1696
    b94[2][82] = 0.1180
    a94[3][82] = 4.5592
    b94[3][82] = 0.0020
    c94[82] = 0.0
    
    vsc = 0.0
    for i in range(4):
        vhlp = b94[i][nhlp] * S * S
        
        if vhlp > 15.0:
            vhlp = 15.0
        vsc += a94[i][nhlp] * math.exp(-vhlp)
    
    vsc += c94[nhlp]
    return vsc

def geo_intensity_3(divided, minTheta, angleLimit, extMinTh, extMaxTh, dA, dB, prefix, divider):
    print("Running algorithm analysis version...")
    
    fSubs = [0.0] * 3
    fBuff = [0.0] * 3
    fA = [0.0] * 3
    fB = [0.0] * 3

    wavelength = 1.5419
    #dSubs = 3.905 / 2.0
    #dBuff = 3.960 / 2.0
    
    pSubs = 1.0
    gSubs = 0.0
    pBuff = 1.0
    gBuff = 0.0
    pA = 1.0
    gA = 0.0
    pB = 1.0
    gB = 0.0
    
    theta = 0.0
    stepTheta = angleLimit / divided
    nSubs = 2 * 10000
    
    stna = 14 * 1.0
    stmb = 9 * 1.0
    snat = 0.6
    nRepeat = 6
    seed = 86456
    nA = 14
    mB = 9
    
    opR = 0.0
    opI = 0.0
    vM = 0.0
    vM2 = 0.0

    angles = []
    intensities = []

    minValue = float('inf')
    maxValue = float('-inf')

    random.seed(seed)
    vjInt = 0.0
    vj2 = 0.0

    fileName = f"res_bas_06_th_{extMinTh}{prefix}{extMaxTh}.txt"

    with open(fileName, "w") as streamWriter:
        for i in range(divided + 1):

            theta = minTheta + i * stepTheta
            thetaR = theta * (math.pi / 180.0)
            S = math.sin(thetaR) / wavelength
            
            #rep term
            dPer = stna * 2.0 * dA + stmb * 2.0 * dB
            vbr = 1.0 - math.cos(4 * math.pi * nRepeat * dPer * S)
            vbi = 0.0 - math.sin(4 * math.pi * nRepeat * dPer * S)
            vdr = 1.0 - math.cos(4 * math.pi * dPer * S)
            vdi = 0.0 - math.sin(4 * math.pi * dPer * S)
            opr = vbr * vdr + vbi * vdi
            opi = -vbr * vdi + vbi * vdr
            hdhd = vdr * vdr + vdi * vdi
            if hdhd != 0.0:
                opr = opr / hdhd
                opi = opi / hdhd
            vjint2 = opr * opr + opi * opi

            fA[0] = atsc(57, S) * 0.67 + atsc(38, S) * 0.33
            fA[1] = atsc(25, S)
            fA[2] = atsc(8, S)
            fB[0] = atsc(56, S)
            fB[1] = atsc(22, S)
            fB[2] = atsc(8, S)

            xBas = 0.0
            sumReal = 0.0
            sumImag = 0.0
            xCurrent = xBas

            for k in range(1, nRepeat + 1):
                            
                for o in range(1, nA + 1):
                    xCurrent += dA
                    vMult = (fA[0] + fA[2]) * pA * math.exp(-gA * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    sumReal += vMult * cContr
                    sumImag += vMult * sContr
                    xCurrent += dA
                    vMult = (fA[1] + 2.0 * fA[2]) * pA * math.exp(-gA * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    sumReal += vMult * cContr
                    sumImag += vMult * sContr

                for l in range(1, mB + 1):
                    xCurrent += dB
                    vMult = (fB[0] + fB[2]) * pB * math.exp(-gB * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    sumReal += vMult * cContr
                    sumImag += vMult * sContr
                    xCurrent += dB
                    vMult = (fB[1] + 2.0 * fB[2]) * pB * math.exp(-gB * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    sumReal += vMult * cContr
                    sumImag += vMult * sContr
                    
                if k == 1:
                    vj2 = sumReal * sumReal + sumImag * sumImag

            vjInt = sumReal * sumReal + sumImag * sumImag
                        
            if math.sin(thetaR) != 0.0:
                vM = 1.0
                vM2 = 1.0
            vjInt *= vM
            vj2 *= vM2

            angles.append(2 * theta)
            intensities.append(vj2 + vM2)

            n = f"{2 * theta:20.10e}{ vM + vjInt:20.10e}" #bas = vjInt + vM / divider fac = vjint2 / divider bl = vj2

            if angles[i] >= extMinTh and angles[i] < extMaxTh:
                streamWriter.write(n + "\n")

            if i == 0:
                minValue = intensities[i]
            if intensities[i] < minValue:
                minValue = intensities[i]
            if intensities[i] > maxValue:
                maxValue = intensities[i]

    return angles, intensities               

def geo_intensity_pto(divided, minTheta, angleLimit, extMinTh, extMaxTh, dA, dB, prefix, divider, select_version, snat):
    print("Running PTO version...")
    
    start = time.time()
    fSubs = [0.0] * 3
    fBuff = [0.0] * 3
    fA = [0.0] * 3
    fB = [0.0] * 3

    wavelength = 1.5406
    dSubs = 3.905 / 2.0
    dBuff = 3.960 / 2.0
    
    pSubs = 1.0
    gSubs = 0.0
    pBuff = 1.0
    gBuff = 0.0
    pA = 1.0
    gA = 0.0
    pB = 1.0
    gB = 0.0
    
    theta = 0.0
    stepTheta = angleLimit / divided
    nSubs = 2 * 10000
    tau = (1.5 * 10000.0)# / 5.0
    vobmi = 1.0 / (1.5 * 10000.0)
    nBuff = 10
    nblk = 1000
    
    stna = 3 * 1.0
    stmb = 3 * 1.0
    
    #stna = 14 * 1.0
    #stmb = 9 * 1.0
    #snat = 0.6
    nRepeat = 10
    seed = 86456
    nA = 3
    mB = 3
    seed1 = 2147483562
    seed2 = 2147483398
    
    opR = 0.0
    opI = 0.0
    vM = 0.0

    angles = []
    intensities = []

    minValue = float('inf')
    maxValue = float('-inf')

    random.seed(seed)
    vjInt = 0.0

    fileName = f"res_pto_06_th_{extMinTh}{prefix}{extMaxTh}.txt"

    with open(fileName, "w") as streamWriter:
        for i in range(divided + 1):
            vjInt = 0.0
            for w in range(2):
                if w == 0:
                    wavelength = 1.5419
                elif w != 0 and select_version == 'ideal1':
                    wavelength = 1.5419
                elif w != 0 and (select_version == 'ideal2' or select_version == 'monte_carlo'):
                    wavelength = 1.5444

                for srand in range(nblk):
                    theta = minTheta + i * stepTheta
                    thetaR = theta * (math.pi / 180.0)
                    S = math.sin(thetaR) / wavelength

                    fSubs[0] = atsc(38, S)
                    fSubs[1] = atsc(22, S)
                    fSubs[2] = atsc(8, S)
                    fBuff[0] = atsc(38, S)
                    fBuff[1] = atsc(44, S)
                    fBuff[2] = atsc(8, S)
                    fA[0] = atsc(82, S)
                    fA[1] = atsc(22, S)
                    fA[2] = atsc(8, S)
                    fB[0] = atsc(38, S)
                    fB[1] = atsc(22, S)
                    fB[2] = atsc(8, S)

                    xBas = 0.0
                    sumReal = 0.0
                    sumImag = 0.0
                    xCurrent = xBas + dSubs
                    
                    if nSubs >= 1:
                        xCurrent = xBas
                        vMult = (fSubs[1] + 2.0 * fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        fsR = vMult * cContr
                        fsI = vMult * sContr

                        xCurrent -= dSubs
                        vMult = (fSubs[0] + fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        fsR += vMult * cContr
                        fsI += vMult * sContr

                        vbR = 1.0
                        vbI = 0.0 
                        
                        if math.sin(thetaR) != 0.0:
                            vobmi = 1.0 / (tau * math.sin(thetaR))

                        if vobmi * nSubs * 2.0 * dSubs >= 15.0:
                            vLact = 0.0
                        else:
                            vLact = math.exp(-vobmi * nSubs * 2.0 * dSubs)

                        vbR -= vLact * math.cos(-4 * math.pi * nSubs * 2.0 * dSubs * S)
                        vbI -= vLact * math.sin(-4 * math.pi * nSubs * 2.0 * dSubs * S)
                        vdR = 1.0
                        vdI = 0.0
                        vHact = math.exp(-vobmi * 2.0 * dSubs)
                        vdR -= vHact * math.cos(-4 * math.pi * 2.0 * dSubs * S)
                        vdI -= vHact * math.sin(-4 * math.pi * 2.0 * dSubs * S)
                        opR = vbR * vdR + vbI * vdI
                        opI = -vbR * vdI + vbI * vdR
                        hdHd = vdR * vdR + vdI * vdI
                        if hdHd != 0.0:
                            opR /= hdHd
                            opI /= hdHd
                        sumReal += fsR * opR - fsI * opI
                        sumImag += fsR * opI + fsI * opR

                    xCurrent = xBas

                    for nB in range(1, nBuff + 1):
                        xCurrent += dBuff
                        vMult = (fBuff[0] + fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr
                        xCurrent += dBuff
                        vMult = (fBuff[1] + 2.0 * fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr

                    for k in range(1, nRepeat + 1):
                        # testing
                        # for g1 in range(2):
                            # ganum = 0.0
                            # for g2 in range(12):
                                # ganum += random.random()
                            # ganum -= 6.0

                            # if g1 == 0:
                                # nA = round(snat * ganum + stna)
                                # if nA < 0:
                                    # nA = 0
                            # else:
                                # mB = round(snat * ganum + stmb)
                                # if mB < 0:
                                    # mB = 0
                        if select_version == 'monte_carlo': 
                            for g1 in range(2):
                                seed = 1#random.random()
                                lcg = LCG(seed)#changed seed
                                z0, z1 = box_muller(lcg)
                                if g1 == 0:
                                    nA = round(snat * z0 + stna) 
                                    if nA < 0:
                                        nA = 0
                                    #print(f"Box nA: {nA}")
                                else:
                                    mB = round(snat * z1 + stmb)
                                    if mB < 0:
                                        mB = 0

                        else:
                            nA = 3
                            mB = 3
                                    
                        for o in range(1, nA + 1):
                            xCurrent += dA
                            vMult = (fA[0] + fA[2]) * pA * math.exp(-gA * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr
                            xCurrent += dA
                            vMult = (fA[1] + 2.0 * fA[2]) * pA * math.exp(-gA * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr

                        for l in range(1, mB + 1):
                            xCurrent += dB
                            vMult = (fB[0] + fB[2]) * pB * math.exp(-gB * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr
                            xCurrent += dB
                            vMult = (fB[1] + 2.0 * fB[2]) * pB * math.exp(-gB * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr

                    #vjInt = sumReal * sumReal + sumImag * sumImag
                    if w == 0:
                        #vjInt = sumReal * sumReal + sumImag * sumImag
                        vjInt += 1.00 * (sumReal * sumReal + sumImag * sumImag)
                    else:
                        #vjInt = sumReal * sumReal + sumImag * sumImag
                        vjInt += 0.52 * (sumReal * sumReal + sumImag * sumImag)
                        
            if math.sin(thetaR) != 0.0:
                vM = 1.0
            vM = 1.0
            vjInt *= vM
            vM = (1.00 / 1.52) * (1.0 / nblk)
            vjInt *= vM

            angles.append(2 * theta)
            intensities.append(vjInt)

            n = f"{2 * theta:20.10e}{vjInt / divider:20.10e}"

            if angles[i] >= extMinTh and angles[i] < extMaxTh:
                streamWriter.write(n + "\n")

            if i == 0:
                minValue = intensities[i]
            if intensities[i] < minValue:
                minValue = intensities[i]
            if intensities[i] > maxValue:
                maxValue = intensities[i]
                
    end = time.time()
    sum = end - start
    return angles, intensities                 

def geo_intensity_lsmo_t(divided, minTheta, angleLimit, extMinTh, extMaxTh, dA, dB, prefix, divider, select_version, snat):
    print("Running multithreaded version...")
    
    fSubs = [0.0] * 3
    fBuff = [0.0] * 3
    fA = [0.0] * 3
    fB = [0.0] * 3

    wavelength = 1.5419
    dSubs = 3.905 / 2.0
    dBuff = 3.960 / 2.0
    
    pSubs = 1.0
    gSubs = 0.0
    pBuff = 1.0
    gBuff = 0.0
    pA = 1.0
    gA = 0.0
    pB = 1.0
    gB = 0.0
    
    theta = 0.0
    stepTheta = angleLimit / divided
    nSubs = 2 * 10000
    tau = (1.5 * 10000.0) / 5.0
    vobmi = 1.0 / (1.5 * 10000.0)
    vjInt = 0.0
    nBuff = 0
    nblk = 1000
    stna = 14 * 1.0
    stmb = 9 * 1.0
    #snat = 0.6
    nRepeat = 6
    
    vM = 0.0
    
    angles = []
    intensities = []

    minValue = float('inf')
    maxValue = float('-inf')

    fileName = f"res_lsmo_06_th_{extMinTh}{prefix}{extMaxTh}.txt"

    def compute_vjint(i, vM):
        vjInt = 0.0
        vobmi = 0.0
        vm = 0.0
        if i % 1000 == 0:
            print(i)
        for w in range(2):
            if w == 0:
                wavelength = 1.5419
            elif w != 0 and select_version == 'ideal1':
                wavelength = 1.5419
            elif w != 0 and (select_version == 'ideal2' or select_version == 'monte_carlo'):
                wavelength = 1.5444

            for srand in range(nblk):
                theta = minTheta + i * stepTheta
                thetaR = theta * (math.pi / 180.0)
                S = math.sin(thetaR) / wavelength

                fSubs[0] = atsc(38, S)
                fSubs[1] = atsc(22, S)
                fSubs[2] = atsc(8, S)
                fBuff[0] = atsc(38, S)
                fBuff[1] = atsc(44, S)
                fBuff[2] = atsc(8, S)
                fA[0] = atsc(57, S) * 0.67 + atsc(38, S) * 0.33
                fA[1] = atsc(25, S)
                fA[2] = atsc(8, S)
                fB[0] = atsc(56, S)
                fB[1] = atsc(22, S)
                fB[2] = atsc(8, S)

                xBas = 0.0
                sumReal = 0.0
                sumImag = 0.0

                if nSubs >= 1:
                    xCurrent = xBas
                    vMult = (fSubs[1] + 2.0 * fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    fsR = vMult * cContr
                    fsI = vMult * sContr

                    xCurrent -= dSubs
                    vMult = (fSubs[0] + fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    fsR += vMult * cContr
                    fsI += vMult * sContr

                    vbR = 1.0
                    vbI = 0.0 

                    if math.sin(thetaR) != 0.0:
                        vobmi = 1.0 / (tau * math.sin(thetaR))

                    if vobmi * nSubs * 2.0 * dSubs >= 15.0:
                        vLact = 0.0
                    else:
                        vLact = math.exp(-vobmi * nSubs * 2.0 * dSubs)

                    vbR -= vLact * math.cos(-4 * math.pi * nSubs * 2.0 * dSubs * S)
                    vbI -= vLact * math.sin(-4 * math.pi * nSubs * 2.0 * dSubs * S)
                    vdR = 1.0
                    vdI = 0.0
                    vHact = math.exp(-vobmi * 2.0 * dSubs)
                    vdR -= vHact * math.cos(-4 * math.pi * 2.0 * dSubs * S)
                    vdI -= vHact * math.sin(-4 * math.pi * 2.0 * dSubs * S)
                    opR = vbR * vdR + vbI * vdI
                    opI = -vbR * vdI + vbI * vdR
                    hdHd = vdR * vdR + vdI * vdI
                    if hdHd != 0.0:
                        opR /= hdHd
                        opI /= hdHd
                    sumReal += fsR * opR - fsI * opI
                    sumImag += fsR * opI + fsI * opR

                xCurrent = xBas

                for nB in range(nBuff + 1):
                    xCurrent += dBuff
                    vMult = (fBuff[0] + fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    sumReal += vMult * cContr
                    sumImag += vMult * sContr
                    xCurrent += dBuff
                    vMult = (fBuff[1] + 2.0 * fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                    cContr = math.cos(4 * math.pi * xCurrent * S)
                    sContr = math.sin(4 * math.pi * xCurrent * S)
                    sumReal += vMult * cContr
                    sumImag += vMult * sContr

                for k in range(1, nRepeat + 1):
                    if select_version == 'monte_carlo':            
                        for g1 in range(2):
                            seed = 1  # random.random()
                            lcg = LCG(seed)
                            z0, z1 = box_muller(lcg)

                            if g1 == 0:
                                nA = round(snat * z0 + stna) 
                                if nA < 0:
                                    nA = 0
                            else:
                                mB = round(snat * z1 + stmb)
                                if mB < 0:
                                    mB = 0
                    else:
                        nA = 14
                        mB = 9

                    for o in range(1, nA + 1):
                        xCurrent += dA
                        vMult = (fA[0] + fA[2]) * pA * math.exp(-gA * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr
                        xCurrent += dA
                        vMult = (fA[1] + 2.0 * fA[2]) * pA * math.exp(-gA * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr

                    for l in range(1, mB + 1):
                        xCurrent += dB
                        vMult = (fB[0] + fB[2]) * pB * math.exp(-gB * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr
                        xCurrent += dB
                        vMult = (fB[1] + 2.0 * fB[2]) * pB * math.exp(-gB * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr

                if w == 0:
                    vjInt += 1.00 * (sumReal * sumReal + sumImag * sumImag)
                else:
                    vjInt += 0.52 * (sumReal * sumReal + sumImag * sumImag)
                    
        if math.sin(thetaR) != 0.0:
            vM = 1.0
          
        vjInt *= vM
        vM = (1.00 / 1.52) * (1.0 / nblk)
        vjInt *= vM

        angles.append(2 * theta)
        intensities.append(vjInt)

        return (i, theta, vjInt)

    def save_results_to_file(results):
        with open(fileName, "w") as streamWriter:
            for i, theta, vjInt in results:
                n = f"{2 * theta:20.10e}{vjInt / divider:20.10e}"
                if angles[i] >= extMinTh and angles[i] < extMaxTh:
                    streamWriter.write(n + "\n")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_vjint, i, vM) for i in range(divided + 1)]
        results = [f.result() for f in futures]

    save_results_to_file(results)

    return angles, intensities

def geo_intensity_lsmo(divided, minTheta, angleLimit, extMinTh, extMaxTh, dA, dB, prefix, divider, select_version, snat):
    
    print("Running single-threaded version...")
    
    fSubs = [0.0] * 3
    fBuff = [0.0] * 3
    fA = [0.0] * 3
    fB = [0.0] * 3

    wavelength = 1.5419
    dSubs = 3.905 / 2.0
    dBuff = 3.960 / 2.0
    
    pSubs = 1.0
    gSubs = 0.0
    pBuff = 1.0
    gBuff = 0.0
    pA = 1.0
    gA = 0.0
    pB = 1.0
    gB = 0.0
    
    theta = 0.0
    stepTheta = angleLimit / divided
    nSubs = 2 * 10000
    tau = (1.5 * 10000.0) / 5.0 #changed 2 / 5.0 #changed
    vobmi = 1.0 / (1.5 * 10000.0)
    nBuff = 0
    nblk = 100
    stna = 14 * 1.0
    stmb = 9 * 1.0
    
    snat2 = 0.0000000006
    nRepeat = 6
    seed1 = 2147483562
    seed2 = 2147483398
    
    opR = 0.0
    opI = 0.0
    vM = 0.0

    angles = []
    intensities = []

    minValue = float('inf')
    maxValue = float('-inf')

    fileName = f"res_lsmo_06_th_{extMinTh}{prefix}{extMaxTh}.txt"

    with open(fileName, "w") as streamWriter:
        for i in range(divided + 1):
            vjInt = 0.0
            for w in range(2):
                if w == 0:
                    wavelength = 1.5419
                elif w != 0 and select_version == 'ideal1':
                    wavelength = 1.5419
                elif w != 0 and (select_version == 'ideal2' or select_version == 'monte_carlo'):
                    wavelength = 1.5444

                for srand in range(nblk):
                    theta = minTheta + i * stepTheta
                    thetaR = theta * (math.pi / 180.0)
                    S = math.sin(thetaR) / wavelength

                    fSubs[0] = atsc(38, S)
                    fSubs[1] = atsc(22, S)
                    fSubs[2] = atsc(8, S)
                    fBuff[0] = atsc(38, S)
                    fBuff[1] = atsc(44, S)
                    fBuff[2] = atsc(8, S)
                    fA[0] = atsc(57, S) * 0.67 + atsc(38, S) * 0.33
                    fA[1] = atsc(25, S)
                    fA[2] = atsc(8, S)
                    fB[0] = atsc(56, S)
                    fB[1] = atsc(22, S)
                    fB[2] = atsc(8, S)

                    xBas = 0.0
                    sumReal = 0.0
                    sumImag = 0.0

                    if nSubs >= 1:
                        xCurrent = xBas
                        vMult = (fSubs[1] + 2.0 * fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        fsR = vMult * cContr
                        fsI = vMult * sContr

                        xCurrent -= dSubs
                        vMult = (fSubs[0] + fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        fsR += vMult * cContr
                        fsI += vMult * sContr

                        vbR = 1.0
                        vbI = 0.0 
                        
                        if math.sin(thetaR) != 0.0:
                            vobmi = 1.0 / (tau * math.sin(thetaR))

                        if vobmi * nSubs * 2.0 * dSubs >= 15.0:
                            vLact = 0.0
                        else:
                            vLact = math.exp(-vobmi * nSubs * 2.0 * dSubs)

                        vbR -= vLact * math.cos(-4 * math.pi * nSubs * 2.0 * dSubs * S)
                        vbI -= vLact * math.sin(-4 * math.pi * nSubs * 2.0 * dSubs * S)
                        vdR = 1.0
                        vdI = 0.0
                        vHact = math.exp(-vobmi * 2.0 * dSubs)
                        vdR -= vHact * math.cos(-4 * math.pi * 2.0 * dSubs * S)
                        vdI -= vHact * math.sin(-4 * math.pi * 2.0 * dSubs * S)
                        opR = vbR * vdR + vbI * vdI
                        opI = -vbR * vdI + vbI * vdR
                        hdHd = vdR * vdR + vdI * vdI
                        if hdHd != 0.0:
                            opR /= hdHd
                            opI /= hdHd
                        sumReal += fsR * opR - fsI * opI
                        sumImag += fsR * opI + fsI * opR

                    xCurrent = xBas

                    for nB in range(nBuff + 1):
                        xCurrent += dBuff
                        vMult = (fBuff[0] + fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr
                        xCurrent += dBuff
                        vMult = (fBuff[1] + 2.0 * fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                        cContr = math.cos(4 * math.pi * xCurrent * S)
                        sContr = math.sin(4 * math.pi * xCurrent * S)
                        sumReal += vMult * cContr
                        sumImag += vMult * sContr

                    for k in range(1, nRepeat + 1):
                        # testing version
                        # for g1 in range(2):
                            # ganum = 0.0
                            # for g2 in range(12):
                                # ganum += random.random()
                            # ganum -= 6.0

                            # if g1 == 0:
                                # nA = round(snat * ganum + stna)
                                # if nA < 0:
                                    # nA = 0
                            # else:
                                # mB = round(snat * ganum + stmb)
                                # if mB < 0:
                                    # mB = 0    
                        if select_version == 'monte_carlo':           
                            for g1 in range(2):
                                seed = 1#random.random()
                                lcg = LCG(seed)
                                z0, z1 = box_muller(lcg)

                                if g1 == 0:
                                    nA = round(snat * z0 + stna) 
                                    if nA < 0:
                                        nA = 0
                                else:
                                    mB = round(snat * z1 + stmb)
                                    if mB < 0:
                                        mB = 0
                        else:
                            nA = 14
                            mB = 9

                        for o in range(1, nA + 1):
                            xCurrent += dA
                            vMult = (fA[0] + fA[2]) * pA * math.exp(-gA * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr
                            xCurrent += dA
                            vMult = (fA[1] + 2.0 * fA[2]) * pA * math.exp(-gA * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr

                        for l in range(1, mB + 1):
                            xCurrent += dB
                            vMult = (fB[0] + fB[2]) * pB * math.exp(-gB * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr
                            xCurrent += dB
                            vMult = (fB[1] + 2.0 * fB[2]) * pB * math.exp(-gB * S * S)
                            cContr = math.cos(4 * math.pi * xCurrent * S)
                            sContr = math.sin(4 * math.pi * xCurrent * S)
                            sumReal += vMult * cContr
                            sumImag += vMult * sContr

                    if w == 0:
                        vjInt += 1.00 * (sumReal * sumReal + sumImag * sumImag)
                    else:
                        vjInt += 0.52 * (sumReal * sumReal + sumImag * sumImag)

            if math.sin(thetaR) != 0.0:
                vM = 1.0
            vjInt *= vM
            vM = (1.00 / 1.52) * (1.0 / nblk)
            vjInt *= vM

            angles.append(2 * theta)
            intensities.append(vjInt)

            n = f"{2 * theta:20.10e}{vjInt / divider:20.10e}"

            if angles[i] >= extMinTh and angles[i] < extMaxTh:
                streamWriter.write(n + "\n")

            if i == 0:
                minValue = intensities[i]
            if intensities[i] < minValue:
                minValue = intensities[i]
            if intensities[i] > maxValue:
                maxValue = intensities[i]

    return angles, intensities

def generate_values(generator, n=10000):
    return [generator.next() for _ in range(n)]

def measure_execution_time(func, repeats=3):
    return timeit.timeit(lambda: func, number=repeats) / repeats

def measure_cpu_time(func):
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    return s.getvalue()

def measure_memory_usage(func):
    mem_usage = memory_usage((func,), interval=0.01, timeout=1)
    return max(mem_usage)

def measure_performance(func, **kwargs):
    process = psutil.Process()
    start_time = time.time()
    start_cpu = time.process_time()
    start_memory = process.memory_info().rss
    
    result = func(**kwargs)
    
    end_cpu = time.process_time()
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    cpu_time = end_cpu - start_cpu
    wall_time = end_time - start_time
    memory_usage = (end_memory - start_memory) / 1024 / 1024
    
    return result, cpu_time, wall_time, memory_usage

def plot_comparison(angles1, intensities1, angles2, intensities2, cpu_times, wall_times, memory_usages):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(angles1, intensities1, label='geo_intensity_lsmo', color='blue')
    plt.plot(angles2, intensities2, label='geo_intensity_lsmo2', color='red', linestyle='--')
    plt.xlabel('Angle (2θ)')
    plt.ylabel('Intensity')
    plt.title('Comparison of Intensities')
    plt.legend()
    
    labels = ['CPU Time', 'Wall Time', 'Memory Usage']
    x = np.arange(len(labels))
    width = 0.35
    
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, [cpu_times[0], wall_times[0], memory_usages[0]], width, label='geo_intensity_lsmo', color='blue')
    plt.bar(x + width/2, [cpu_times[1], wall_times[1], memory_usages[1]], width, label='geo_intensity_lsmo2', color='red')
    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('Resource Usage Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_diffraction(func1, func2, kwargs1, kwargs2):
    result1, cpu_time1, wall_time1, memory_usage1 = measure_performance(func1, **kwargs1)
    angles1, intensities1 = result1
    
    result2, cpu_time2, wall_time2, memory_usage2 = measure_performance(func2, **kwargs2)
    angles2, intensities2 = result2
    
    print("geo_intensity_lsmo:")
    print(f"CPU Time: {cpu_time1:.4f} s")
    print(f"Wall Time: {wall_time1:.4f} s")
    print(f"Memory Usage: {memory_usage1:.2f} MB")
    
    print("\ngeo_intensity_lsmo2:")
    print(f"CPU Time: {cpu_time2:.4f} s")
    print(f"Wall Time: {wall_time2:.4f} s")
    print(f"Memory Usage: {memory_usage2:.2f} MB")
    
    plot_comparison(angles1, intensities1, angles2, intensities2, 
                    [cpu_time1, cpu_time2], [wall_time1, wall_time2], [memory_usage1, memory_usage2])

def str_to_float(value):
    try:
        if '/' in value:
            numerator, denominator = value.split('/')
            return float(numerator) / float(denominator)
        return float(value)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid number or division expression")

def main():
    parser = argparse.ArgumentParser(description="X-ray diffraction analysis with multiple use cases")
    
    parser.add_argument('--use_case', type=int, required=True, choices=[1, 2, 3, 4, 5],
                       help='Select use case (1-5)')
    
    # Common parameters
    parser.add_argument('--divided', type=int, default=3600, help='Number of divisions')
    parser.add_argument('--minTheta', type=float, default=0.0, help='Minimum theta angle')
    parser.add_argument('--angleLimit', type=float, default=90.0, help='Angle limit')
    parser.add_argument('--divider', type=int, default=1, help='Divider value')
    parser.add_argument('--threaded', action='store_true', 
                       help='Enable threaded performance analysis using analyze_diffraction')
    
    # Parameters for specific use cases
    parser.add_argument('--dA_values', nargs='+', type=str_to_float, 
                       help='dA values (e.g., "3.876/2.0 3.860/2.0")')
    parser.add_argument('--dB_values', nargs='+', type=str_to_float, 
                       help='dB values (e.g., "4.036/2.0 4.100/2.0")')
    parser.add_argument('--prefixes', nargs='+', type=str, 
                       help='Prefixes for output files')
    parser.add_argument('--calc_types', nargs='+', type=str, 
                       help='Calculation types (ideal1, ideal2, monte_carlo)')
    parser.add_argument('--snat_values', nargs='+', type=float, 
                       help='SNAT values for Monte Carlo')
    
    args = parser.parse_args()

    if args.use_case == 1:
        # Use Case 1: 30-80 range with different calculation types
        minTh, maxTh = 30, 80
        args.dA_values = args.dA_values or [3.876/2.0]
        args.dB_values = args.dB_values or [4.036/2.0]
        args.prefixes = args.prefixes or ["_1_", "_2_", "_3_"]
        args.calc_types = args.calc_types or ["ideal1", "ideal2", "monte_carlo"]
        args.snat_values = args.snat_values or [0.0, 0.0, 0.6]
        
        if args.threaded:
            print("\nRunning threaded performance comparisons...")
            
            # Compare ideal1 vs ideal2
            print("\nComparison 1: ideal1 vs ideal2")
            kwargs1 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[0],
                'dB': args.dB_values[0],
                'prefix': args.prefixes[0],
                'divider': args.divider,
                'select_version': args.calc_types[0],
                'snat': args.snat_values[0]
            }
            kwargs2 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[0],
                'dB': args.dB_values[0],
                'prefix': args.prefixes[1],
                'divider': args.divider,
                'select_version': args.calc_types[1],
                'snat': args.snat_values[1]
            }
            analyze_diffraction(geo_intensity_lsmo, geo_intensity_lsmo, kwargs1, kwargs2)
            
            # Compare ideal2 vs monte_carlo
            print("\nComparison 2: ideal2 vs monte_carlo")
            kwargs1 = kwargs2
            kwargs2 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[0],
                'dB': args.dB_values[0],
                'prefix': args.prefixes[2],
                'divider': args.divider,
                'select_version': args.calc_types[2],
                'snat': args.snat_values[2]
            }
            analyze_diffraction(geo_intensity_lsmo, geo_intensity_lsmo_t, kwargs1, kwargs2)
        else:
            print("\nRunning single-threaded calculations sequentially...")
            for prefix, calc_type, snat in zip(args.prefixes, args.calc_types, args.snat_values):
                kwargs = {
                    'divided': args.divided,
                    'minTheta': args.minTheta,
                    'angleLimit': args.angleLimit,
                    'extMinTh': minTh,
                    'extMaxTh': maxTh,
                    'dA': args.dA_values[0],
                    'dB': args.dB_values[0],
                    'prefix': prefix,
                    'divider': args.divider,
                    'select_version': calc_type,
                    'snat': snat
                }
                print(f"\nRunning {calc_type} version...")

                geo_intensity_lsmo(**kwargs)
        
    elif args.use_case == 2:
        # Use Case 2: 60-80 range with different lattice parameters
        minTh, maxTh = 60, 80
        args.dA_values = args.dA_values or [3.876/2.0, 3.860/2.0]
        args.dB_values = args.dB_values or [4.036/2.0, 4.100/2.0]
        args.prefixes = args.prefixes or ["_60_", "_80_"]
        args.calc_types = args.calc_types or ["monte_carlo", "monte_carlo"]
        args.snat_values = args.snat_values or [0.3, 0.6]
        
        if args.threaded:
            print("\nRunning threaded performance comparison...")
            kwargs1 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[0],
                'dB': args.dB_values[0],
                'prefix': args.prefixes[0],
                'divider': args.divider,
                'select_version': args.calc_types[0],
                'snat': args.snat_values[0]
            }
            kwargs2 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[1],
                'dB': args.dB_values[1],
                'prefix': args.prefixes[1],
                'divider': args.divider,
                'select_version': args.calc_types[1],
                'snat': args.snat_values[1]
            }
            analyze_diffraction(geo_intensity_lsmo_t, geo_intensity_lsmo_t, kwargs1, kwargs2)
        else:
            print("\nRunning single-threaded calculations sequentially...")
            for dA, dB, prefix, calc_type, snat in zip(args.dA_values, args.dB_values, 
                                                     args.prefixes, args.calc_types, args.snat_values):
                kwargs = {
                    'divided': args.divided,
                    'minTheta': args.minTheta,
                    'angleLimit': args.angleLimit,
                    'extMinTh': minTh,
                    'extMaxTh': maxTh,
                    'dA': dA,
                    'dB': dB,
                    'prefix': prefix,
                    'divider': args.divider,
                    'select_version': calc_type,
                    'snat': snat
                }
                print(f"\nRunning calculation with dA={dA}, dB={dB}...")
                geo_intensity_lsmo_t(**kwargs)
        
    elif args.use_case == 3:
        # Use Case 3: 10-120 range with different lattice parameters
        minTh, maxTh = 10, 120
        args.dA_values = args.dA_values or [3.860/2.0, 3.876/2.0]
        args.dB_values = args.dB_values or [4.100/2.0, 4.036/2.0]
        args.prefixes = args.prefixes or ["_b_", "_"]
        args.calc_types = args.calc_types or ["monte_carlo", "monte_carlo"]
        args.snat_values = args.snat_values or [0.3, 0.6]
        
        if args.threaded:
            print("\nRunning threaded performance comparison...")
            kwargs1 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[0],
                'dB': args.dB_values[0],
                'prefix': args.prefixes[0],
                'divider': args.divider,
                'select_version': args.calc_types[0],
                'snat': args.snat_values[0]
            }
            kwargs2 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': args.dA_values[1],
                'dB': args.dB_values[1],
                'prefix': args.prefixes[1],
                'divider': args.divider,
                'select_version': args.calc_types[1],
                'snat': args.snat_values[1]
            }
            analyze_diffraction(geo_intensity_lsmo_t, geo_intensity_lsmo_t, kwargs1, kwargs2)
        else:
            print("\nRunning single-threaded calculations sequentially...")
            for dA, dB, prefix, calc_type, snat in zip(args.dA_values, args.dB_values, 
                                                     args.prefixes, args.calc_types, args.snat_values):
                kwargs = {
                    'divided': args.divided,
                    'minTheta': args.minTheta,
                    'angleLimit': args.angleLimit,
                    'extMinTh': minTh,
                    'extMaxTh': maxTh,
                    'dA': dA,
                    'dB': dB,
                    'prefix': prefix,
                    'divider': args.divider,
                    'select_version': calc_type,
                    'snat': snat
                }
                print(f"\nRunning calculation with dA={dA}, dB={dB}...")
                geo_intensity_lsmo_t(**kwargs)
        
    elif args.use_case == 4:
        # Use Case 4: Special case for geo_intensity_3
        minTh, maxTh = 40, 50
        kwargs = {
            'divided': args.divided,
            'minTheta': args.minTheta,
            'angleLimit': args.angleLimit,
            'extMinTh': minTh,
            'extMaxTh': maxTh,
            'dA': 3.876/2.0,
            'dB': 4.036/2.0,
            'prefix': "_",
            'divider': args.divider
        }
        
        print("\nRunning geo_intensity_3...")
        geo_intensity_3(**kwargs)
        plot_xray_spectra_3(minTh, maxTh)
        return
        
    elif args.use_case == 5:
        # Use Case 5: PTO special case
        minTh, maxTh = 10, 60
        dA_pto, dB_pto = 4.145/2.0, 3.905/2.0
        
        if args.threaded:
            print("\nRunning threaded performance comparison...")
            kwargs1 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': dA_pto,
                'dB': dB_pto,
                'prefix': "__",
                'divider': args.divider,
                'select_version': "ideal1",
                'snat': 0.0
            }
            kwargs2 = {
                'divided': args.divided,
                'minTheta': args.minTheta,
                'angleLimit': args.angleLimit,
                'extMinTh': minTh,
                'extMaxTh': maxTh,
                'dA': dA_pto,
                'dB': dB_pto,
                'prefix': "_w_",
                'divider': args.divider,
                'select_version': "ideal2",
                'snat': 0.0
            }
            analyze_diffraction(geo_intensity_pto, geo_intensity_pto, kwargs1, kwargs2)
        else:
            print("\nRunning single-threaded calculations sequentially...")
            geo_intensity_pto(
                divided=args.divided,
                minTheta=args.minTheta,
                angleLimit=args.angleLimit,
                extMinTh=minTh,
                extMaxTh=maxTh,
                dA=dA_pto,
                dB=dB_pto,
                prefix="__",
                divider=args.divider,
                select_version="ideal1",
                snat=0.0
            )
            geo_intensity_pto(
                divided=args.divided,
                minTheta=args.minTheta,
                angleLimit=args.angleLimit,
                extMinTh=minTh,
                extMaxTh=maxTh,
                dA=dA_pto,
                dB=dB_pto,
                prefix="_w__",
                divider=args.divider,
                select_version="ideal2",
                snat=0.0
            )
            geo_intensity_pto(
                divided=args.divided,
                minTheta=args.minTheta,
                angleLimit=args.angleLimit,
                extMinTh=minTh,
                extMaxTh=maxTh,
                dA=dA_pto,
                dB=dB_pto,
                prefix="_m__",
                divider=args.divider,
                select_version="monte_carlo",
                snat=args.snat_values[0]
            )
        
        plot_xray_spectra_pto(minTh, maxTh)
        return

    plot_xray_spectra(
        minTheta=minTh,
        maxTheta=maxTh,
        data_config=f"{minTh}-{maxTh}",
        peaks=(args.use_case == 2)  # Show peaks only for 60-80 range
    )

if __name__ == "__main__":
    main()
