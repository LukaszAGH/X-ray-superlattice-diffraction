import math
import random
import time
import matplotlib.pyplot as plt
import psutil
import numpy as np
import threading
import os
import argparse
import sys

# Global dictionary to store atomic data
atomic_data = {}

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

def load_atomic_data(filename='atomic_data.txt'):
    global atomic_data
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Atomic data file {filename} not found")

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            symbol = parts[0]
            data = {
                'Z': int(parts[1]),
                'a': [float(x) for x in parts[2::2]],
                'b': [float(x) for x in parts[3::2]],
                'c': float(parts[-1])
            }
            atomic_data[symbol] = data
            atomic_data[data['Z']] = data #atomic number

def atsc(element, S):
    if isinstance(element, str):
        data = atomic_data.get(element)
    else:
        data = atomic_data.get(element)

    if not data:
        raise ValueError(f"Element {element} not found in atomic data")

    vsc = 0.0
    for i in range(4):
        vhlp = data['b'][i] * S * S
        if vhlp > 15.0:
            vhlp = 15.0
        vsc += data['a'][i] * math.exp(-vhlp)

    vsc += data['c']
    return vsc

def monitor_resources(exec_times, cpu_usages, memory_usages, stop_event):
    while not stop_event.is_set():
        exec_times.append(time.time())
        cpu_usages.append(psutil.cpu_percent())
        memory_usages.append(psutil.virtual_memory().used / (1024 * 1024))  # MB
        time.sleep(0.1) #every 0.1s

def geo_intensity(divided, minTheta, angleLimit, extMinTh, extMaxTh, dA, dB, divider, select_version, dSubs, dBuff, nSubs, nBuff, nA, mB, nRepeat, buffer_enabled=True, LSMO=True, fSubs1='Sr', fSubs2='Ti', fSubs3='O', fBuff1='Sr', fBuff2='Ru', fBuff3='O', fA1a='La', fA1b='Sr', fA2='Mn', fA3='O', fB1='Ba', fB2='Ti', fB3='O'):

    fSubs = [0.0] * 3
    fBuff = [0.0] * 3
    fA = [0.0] * 3
    fB = [0.0] * 3

    wavelength = 1.5419

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
    tau = (1.5 * 10000.0) / 5.0
    vobmi = 1.0 / (1.5 * 10000.0)
    nblk = 100
    stna = nA * 1.0
    stmb = mB * 1.0
    snat = 0.6

    snat2 = 0.0000000006
    seed1 = 2147483562
    seed2 = 2147483398

    opR = 0.0
    opI = 0.0
    vM = 0.0

    angles = []
    intensities = []

    minValue = float('inf')
    maxValue = float('-inf')

    fileName = f"res_lsmo_06_th_{extMinTh}_{select_version}_{extMaxTh}.txt"

    exec_times = []
    cpu_usages = []
    memory_usages = []

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(exec_times, cpu_usages, memory_usages, stop_event))
    monitor_thread.start()

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

                    fSubs[0] = atsc(fSubs1, S)
                    fSubs[1] = atsc(fSubs2, S)
                    fSubs[2] = atsc(fSubs3, S)

                    if buffer_enabled:
                        fBuff[0] = atsc(fBuff1, S)
                        fBuff[1] = atsc(fBuff2, S)
                        fBuff[2] = atsc(fBuff3, S)

                    if LSMO:
                        fA[0] = atsc(fA1a, S) * 0.67 + atsc(fA1b, S) * 0.33
                    else:
                        fA[0] = atsc(fA1a, S)  # Assuming fA1a is the only component when not LSMO

                    fA[1] = atsc(fA2, S)
                    fA[2] = atsc(fA3, S)
                    fB[0] = atsc(fB1, S)
                    fB[1] = atsc(fB2, S)
                    fB[2] = atsc(fB3, S)

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

                    if buffer_enabled:
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

    stop_event.set()
    monitor_thread.join()

    return angles, intensities, exec_times, cpu_usages, memory_usages, divided, extMinTh, extMaxTh,  dA, dB, select_version

def analyze_diffraction(func, kwargs):
    versions = kwargs['select_version'].split()

    ordered_versions = []
    if 'ideal1' in versions:
        ordered_versions.append('ideal1')
    if 'ideal2' in versions:
        ordered_versions.append('ideal2')
    if 'monte_carlo' in versions:
        ordered_versions.append('monte_carlo')

    fig_height = 8 + 2 * len(ordered_versions)
    plt.figure(figsize=(15, fig_height))

    grid_rows = 3 + len(ordered_versions)
    gs = plt.GridSpec(grid_rows, 3, height_ratios=[2]*len(ordered_versions) + [1, 1, 1])

    ax1 = plt.subplot(gs[:len(ordered_versions), :])

    colors = ['orange', 'blue', 'green']
    multipliers = [1.0, 0.0001, 0.0000001]

    for i, version in enumerate(ordered_versions):
        current_kwargs = kwargs.copy()
        current_kwargs['select_version'] = version

        angles, intensities, exec_times, cpu_usages, memory_usages, divided, extMinTh, extMaxTh, dA, dB, _ = func(**current_kwargs)

        filtered_angles = []
        filtered_intensities = []
        for angle, intensity in zip(angles, intensities):
            if extMinTh <= angle <= extMaxTh:
                filtered_angles.append(angle)
                filtered_intensities.append(intensity * multipliers[i])

        ax1.semilogy(filtered_angles, filtered_intensities,
                     color=colors[i],
                     linewidth=1.5,
                     label=f'{version} (x{multipliers[i]})')

    ax1.set_xlabel("2θ [°]", fontsize=12)
    ax1.set_ylabel("Intensity [arb. units]", fontsize=12)
    ax1.set_title("Diffraction Pattern Comparison (Log Scale)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.6)
    legend1 = ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                         framealpha=0.9, title="Spectra")

    params_text = (f"Parameters:\n"
                   f"Divided: {divided}\n"
                   f"dA: {dA:.4f}\n"
                   f"dB: {dB:.4f}\n"
                   f"Range: [{extMinTh}, {extMaxTh}]°")

    param_legend = ax1.annotate(params_text,
                                xy=(0.98, 0.75),
                                xycoords='axes fraction',
                                fontsize=10,
                                ha='right',
                                va='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if ordered_versions:
        version = ordered_versions[0]
        _, _, exec_times, cpu_usages, memory_usages, _, _, _, _, _, _ = func(**{**kwargs, 'select_version': version})

        ax2 = plt.subplot(gs[-3, 0])
        ax2.plot(exec_times, color="blue")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Execution Time (s)")
        ax2.set_title("Execution Time")
        ax2.grid(True, linestyle="--", alpha=0.3)

        ax3 = plt.subplot(gs[-3, 1])
        ax3.plot(cpu_usages, color="red")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("CPU Usage (%)")
        ax3.set_title("CPU Usage")
        ax3.grid(True, linestyle="--", alpha=0.3)

        ax4 = plt.subplot(gs[-3, 2])
        ax4.plot(memory_usages, color="green")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Memory Usage (MB)")
        ax4.set_title("Memory Usage")
        ax4.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate diffraction patterns')

    parser.add_argument('--divided', type=int, default=3600, help='Number of divisions')
    parser.add_argument('--minTheta', type=float, default=0.0, help='Minimum theta angle')
    parser.add_argument('--angleLimit', type=float, default=90.0, help='Maximum theta angle')
    parser.add_argument('--extMinTh', type=float, default=30.0, help='Minimum displayed angle')
    parser.add_argument('--extMaxTh', type=float, default=60.0, help='Maximum displayed angle')
    parser.add_argument('--dA', type=float, default=1.938, help='dA parameter')
    parser.add_argument('--dB', type=float, default=2.018, help='dB parameter')
    parser.add_argument('--divider', type=int, default=1, help='Divider value')
    parser.add_argument('--select_version', type=str, required=True, help='Calculation version (ideal1/ideal2/monte_carlo)')

    parser.add_argument('--dSubs', type=float, default=1.9525, help='Substrate d-spacing')
    parser.add_argument('--dBuff', type=float, default=1.98, help='Buffer d-spacing')
    parser.add_argument('--nSubs', type=int, default=20000, help='Number of substrate layers')
    parser.add_argument('--nBuff', type=int, default=1, help='Number of buffer layers')
    parser.add_argument('--nA', type=int, default=14, help='Number of A-site layers')
    parser.add_argument('--mB', type=int, default=9, help='Number of B-site layers')
    parser.add_argument('--nRepeat', type=int, default=6, help='Number of repeats')

    parser.add_argument('--buffer_enabled', type=str2bool, default=True, help='Enable buffer layer')
    parser.add_argument('--LSMO', type=str2bool, default=True, help='Use LSMO composition')

    parser.add_argument('--fSubs1', type=str, default='Sr', help='Substrate element 1')
    parser.add_argument('--fSubs2', type=str, default='Ti', help='Substrate element 2')
    parser.add_argument('--fSubs3', type=str, default='O', help='Substrate element 3')
    parser.add_argument('--fBuff1', type=str, default='Sr', help='Buffer element 1')
    parser.add_argument('--fBuff2', type=str, default='Ru', help='Buffer element 2')
    parser.add_argument('--fBuff3', type=str, default='O', help='Buffer element 3')
    parser.add_argument('--fA1a', type=str, default='La', help='A-site element 1a')
    parser.add_argument('--fA1b', type=str, default='Sr', help='A-site element 1b')
    parser.add_argument('--fA2', type=str, default='Mn', help='A-site element 2')
    parser.add_argument('--fA3', type=str, default='O', help='A-site element 3')
    parser.add_argument('--fB1', type=str, default='Ba', help='B-site element 1')
    parser.add_argument('--fB2', type=str, default='Ti', help='B-site element 2')
    parser.add_argument('--fB3', type=str, default='O', help='B-site element 3')

    return parser.parse_args()

def validate_versions(versions):
    valid_versions = {'ideal1', 'ideal2', 'monte_carlo'}
    versions = versions.split()
    for version in versions:
        if version not in valid_versions:
            raise ValueError(f"Invalid version: {version}. Choose from {valid_versions}")
    return versions

def main():
    load_atomic_data()

    args = parse_arguments()
    versions = validate_versions(args.select_version)

    kwargs = {
        'buffer_enabled': args.buffer_enabled,
        'fSubs1': args.fSubs1,
        'fSubs2': args.fSubs2,
        'fSubs3': args.fSubs3,
        'fBuff1': args.fBuff1,
        'fBuff2': args.fBuff2,
        'fBuff3': args.fBuff3,
        'LSMO': args.LSMO,
        'fA1a': args.fA1a,
        'fA1b': args.fA1b,
        'fA2': args.fA2,
        'fA3': args.fA3,
        'fB1': args.fB1,
        'fB2': args.fB2,
        'fB3': args.fB3,
        'divided': args.divided,
        'minTheta': args.minTheta,
        'angleLimit': args.angleLimit,
        'extMinTh': args.extMinTh,
        'extMaxTh': args.extMaxTh,
        'dA': args.dA,
        'dB': args.dB,
        'dSubs': args.dSubs,
        'dBuff': args.dBuff,
        'nSubs': args.nSubs,
        'nBuff': args.nBuff,
        'nA': args.nA,
        'mB': args.mB,
        'nRepeat': args.nRepeat,
        'divider': args.divider,
        'select_version': args.select_version
    }

    print("Analyzing geo_intensity with parameters:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

    print("\nStarting calculation...")
    analyze_diffraction(geo_intensity, kwargs)

if __name__ == "__main__":
    main()
