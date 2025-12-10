import math
import time
import matplotlib.pyplot as plt
import psutil
import numpy as np
import threading
import os
import argparse
import sys
from numba import cuda

# Global dictionary for host-side loading
atomic_data = {}

def load_atomic_data(filename):
    if not os.path.exists(filename):
        parent_path = os.path.join(os.path.dirname(__file__), '..', filename)
        if os.path.exists(parent_path):
            filename = parent_path
        else:
            raise FileNotFoundError(f"Atomic data file {filename} not found in current or parent directory.")

    atomic_data.clear()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            symbol = parts[0]
            coeffs = []
            for i in range(4):
                coeffs.append(float(parts[2 + i*2])) # a
            for i in range(4):
                coeffs.append(float(parts[3 + i*2])) # b
            coeffs.append(float(parts[-1])) # c

            data = {
                'Z': int(parts[1]),
                'coeffs': np.array(coeffs, dtype=np.float64)
            }
            atomic_data[symbol] = data

def get_element_coeffs(element_symbol):
    if element_symbol not in atomic_data:
        load_atomic_data('atomic_data.txt')

    if element_symbol not in atomic_data:
        raise ValueError(f"Element {element_symbol} missing from data")
    return atomic_data[element_symbol]['coeffs']

@cuda.jit(device=True)
def gpu_atsc(coeffs, S):
    vsc = 0.0
    for i in range(4):
        b_val = coeffs[4 + i]
        a_val = coeffs[i]

        vhlp = b_val * S * S
        if vhlp > 15.0:
            vhlp = 15.0
        vsc += a_val * math.exp(-vhlp)

    vsc += coeffs[8] # c
    return vsc

@cuda.jit(device=True)
def gpu_lcg_next(seed):
    # FIXED: Hardcoded constants to avoid Numba signature mismatch with default args
    a = 1664525
    c = 1013904223
    m = 4294967296 # 2**32

    new_seed = (a * int(seed) + c) % m
    return new_seed, new_seed / m

@cuda.jit(device=True)
def gpu_box_muller(seed):
    s1, u1 = gpu_lcg_next(seed)
    s2, u2 = gpu_lcg_next(s1)

    if u1 < 1e-9: u1 = 1e-9

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
    return z0, z1, s2

@cuda.jit
def calculate_spectrum_kernel(minTheta, stepTheta, select_version_id,
                              coeffs_Subs, coeffs_Buff, coeffs_A, coeffs_B,
                              params_f, params_i, params_b,
                              results_angles, results_intensities):

    idx = cuda.grid(1)
    if idx >= results_intensities.shape[0]:
        return

    dA = params_f[0]; dB = params_f[1]; dSubs = params_f[2]; dBuff = params_f[3]
    snat = params_f[4]; stna = params_f[5]; stmb = params_f[6]

    nSubs = params_i[0]; nBuff = params_i[1]; nA_base = params_i[2]; mB_base = params_i[3]
    nRepeat = params_i[4]; nblk = params_i[5]; LSMO = params_i[6]

    buffer_enabled = params_b[0]

    tau = (1.5 * 10000.0) / 5.0

    theta = minTheta + idx * stepTheta
    thetaR = theta * (math.pi / 180.0)

    vjInt = 0.0

    fSubs = cuda.local.array(3, dtype=np.float64)
    fBuff = cuda.local.array(3, dtype=np.float64)
    fA = cuda.local.array(3, dtype=np.float64)
    fB = cuda.local.array(3, dtype=np.float64)

    pSubs = 1.0
    gSubs = 0.0
    pBuff = 1.0
    gBuff = 0.0
    pA = 1.0
    gA = 0.0
    pB = 1.0
    gB = 0.0

    for w in range(2):
        wavelength = 1.5419
        if w != 0:
            if select_version_id == 0: # ideal1
                wavelength = 1.5419
            else: # ideal2 (1) or monte_carlo (2)
                wavelength = 1.5444

        S = math.sin(thetaR) / wavelength

        fSubs[0] = gpu_atsc(coeffs_Subs[0], S)
        fSubs[1] = gpu_atsc(coeffs_Subs[1], S)
        fSubs[2] = gpu_atsc(coeffs_Subs[2], S)

        if buffer_enabled:
            fBuff[0] = gpu_atsc(coeffs_Buff[0], S)
            fBuff[1] = gpu_atsc(coeffs_Buff[1], S)
            fBuff[2] = gpu_atsc(coeffs_Buff[2], S)

        val_A1a = gpu_atsc(coeffs_A[0], S)
        if LSMO == 1:
            val_A1b = gpu_atsc(coeffs_A[1], S)
            fA[0] = val_A1a * 0.67 + val_A1b * 0.33
        else:
            fA[0] = val_A1a

        fA[1] = gpu_atsc(coeffs_A[2], S)
        fA[2] = gpu_atsc(coeffs_A[3], S)

        fB[0] = gpu_atsc(coeffs_B[0], S)
        fB[1] = gpu_atsc(coeffs_B[1], S)
        fB[2] = gpu_atsc(coeffs_B[2], S)

        for srand in range(nblk):
            xBas = 0.0
            sumReal = 0.0
            sumImag = 0.0

            if nSubs >= 1:
                xCurrent = xBas
                vMult = (fSubs[1] + 2.0 * fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                arg = 4 * math.pi * xCurrent * S
                sumReal += vMult * math.cos(arg)
                sumImag += vMult * math.sin(arg)

                xCurrent -= dSubs
                vMult = (fSubs[0] + fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                arg = 4 * math.pi * xCurrent * S
                sumReal += vMult * math.cos(arg)
                sumImag += vMult * math.sin(arg)

                vobmi = 0.0
                if math.sin(thetaR) != 0.0:
                    vobmi = 1.0 / (tau * math.sin(thetaR))

                vLact = 0.0
                if vobmi * nSubs * 2.0 * dSubs < 15.0:
                    vLact = math.exp(-vobmi * nSubs * 2.0 * dSubs)

                arg_subs = -4 * math.pi * nSubs * 2.0 * dSubs * S
                vbR = 1.0 - vLact * math.cos(arg_subs)
                vbI = 0.0 - vLact * math.sin(arg_subs)

                vHact = math.exp(-vobmi * 2.0 * dSubs)
                arg_unit = -4 * math.pi * 2.0 * dSubs * S
                vdR = 1.0 - vHact * math.cos(arg_unit)
                vdI = 0.0 - vHact * math.sin(arg_unit)

                hdHd = vdR * vdR + vdI * vdI
                opR = 0.0; opI = 0.0
                if hdHd != 0.0:
                    opR = (vbR * vdR + vbI * vdI) / hdHd
                    opI = (-vbR * vdI + vbI * vdR) / hdHd

                xC_tmp = xBas
                vM1 = (fSubs[1] + 2.0 * fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                fsR = vM1 * math.cos(4 * math.pi * xC_tmp * S)
                fsI = vM1 * math.sin(4 * math.pi * xC_tmp * S)

                xC_tmp -= dSubs
                vM2 = (fSubs[0] + fSubs[2]) * pSubs * math.exp(-gSubs * S * S)
                fsR += vM2 * math.cos(4 * math.pi * xC_tmp * S)
                fsI += vM2 * math.sin(4 * math.pi * xC_tmp * S)

                sumReal = fsR * opR - fsI * opI
                sumImag = fsR * opI + fsI * opR

            xCurrent = xBas

            if buffer_enabled == 1:
                for nB_idx in range(nBuff + 1):
                    xCurrent += dBuff
                    vMult = (fBuff[0] + fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                    arg = 4 * math.pi * xCurrent * S
                    sumReal += vMult * math.cos(arg)
                    sumImag += vMult * math.sin(arg)

                    xCurrent += dBuff
                    vMult = (fBuff[1] + 2.0 * fSubs[2]) * pBuff * math.exp(-gBuff * S * S)
                    arg = 4 * math.pi * xCurrent * S
                    sumReal += vMult * math.cos(arg)
                    sumImag += vMult * math.sin(arg)

            nA_curr = nA_base
            mB_curr = mB_base

            for k in range(1, nRepeat + 1):
                if select_version_id == 2: # monte_carlo
                    seed_curr = 1
                    z0, z1, seed_curr = gpu_box_muller(seed_curr)

                    nA_curr = int(snat * z0 + stna + 0.5) # round
                    if nA_curr < 0: nA_curr = 0

                    mB_curr = int(snat * z1 + stmb + 0.5)
                    if mB_curr < 0: mB_curr = 0

                # Layer A
                for o in range(1, nA_curr + 1):
                    xCurrent += dA
                    vMult = (fA[0] + fA[2]) * pA * math.exp(-gA * S * S)
                    arg = 4 * math.pi * xCurrent * S
                    sumReal += vMult * math.cos(arg)
                    sumImag += vMult * math.sin(arg)

                    xCurrent += dA
                    vMult = (fA[1] + 2.0 * fA[2]) * pA * math.exp(-gA * S * S)
                    arg = 4 * math.pi * xCurrent * S
                    sumReal += vMult * math.cos(arg)
                    sumImag += vMult * math.sin(arg)

                # Layer B
                for l in range(1, mB_curr + 1):
                    xCurrent += dB
                    vMult = (fB[0] + fB[2]) * pB * math.exp(-gB * S * S)
                    arg = 4 * math.pi * xCurrent * S
                    sumReal += vMult * math.cos(arg)
                    sumImag += vMult * math.sin(arg)

                    xCurrent += dB
                    vMult = (fB[1] + 2.0 * fB[2]) * pB * math.exp(-gB * S * S)
                    arg = 4 * math.pi * xCurrent * S
                    sumReal += vMult * math.cos(arg)
                    sumImag += vMult * math.sin(arg)

            if w == 0:
                vjInt += 1.00 * (sumReal * sumReal + sumImag * sumImag)
            else:
                vjInt += 0.52 * (sumReal * sumReal + sumImag * sumImag)

    vM = 0.0
    if math.sin(thetaR) != 0.0:
        vM = 1.0
    vjInt *= vM
    vM = (1.00 / 1.52) * (1.0 / nblk)
    vjInt *= vM

    results_angles[idx] = 2 * theta
    results_intensities[idx] = vjInt

def monitor_resources(exec_times, cpu_usages, memory_usages, stop_event):
    while not stop_event.is_set():
        exec_times.append(time.time())
        cpu_usages.append(psutil.cpu_percent())
        memory_usages.append(psutil.virtual_memory().used / (1024 * 1024))
        time.sleep(0.1)

def geo_intensity_cuda(divided, minTheta, angleLimit, extMinTh, extMaxTh, dA, dB, divider, select_version, dSubs, dBuff, nSubs, nBuff, nA, mB, nRepeat, buffer_enabled=True, LSMO=True, fSubs1='Sr', fSubs2='Ti', fSubs3='O', fBuff1='Sr', fBuff2='Ti', fBuff3='O', fA1a='La', fA1b='Sr', fA2='Mn', fA3='O', fB1='Ba', fB2='Ti', fB3='O'):

    load_atomic_data('atomic_data.txt')

    ver_map = {'ideal1': 0, 'ideal2': 1, 'monte_carlo': 2}
    select_version_id = ver_map[select_version]

    c_subs = np.array([get_element_coeffs(fSubs1), get_element_coeffs(fSubs2), get_element_coeffs(fSubs3)], dtype=np.float64)
    c_buff = np.array([get_element_coeffs(fBuff1), get_element_coeffs(fBuff2), get_element_coeffs(fBuff3)], dtype=np.float64)

    c_A = np.array([get_element_coeffs(fA1a), get_element_coeffs(fA1b), get_element_coeffs(fA2), get_element_coeffs(fA3)], dtype=np.float64)

    c_B = np.array([get_element_coeffs(fB1), get_element_coeffs(fB2), get_element_coeffs(fB3)], dtype=np.float64)

    params_f = np.array([dA, dB, dSubs, dBuff, 0.6, float(nA), float(mB)], dtype=np.float64) # 0.6 snat
    params_i = np.array([nSubs, nBuff, nA, mB, nRepeat, 100, 1 if LSMO else 0], dtype=np.int32) # 100 nblk
    params_b = np.array([1 if buffer_enabled else 0], dtype=np.int32)

    total_points = divided + 1
    d_angles = cuda.device_array(total_points, dtype=np.float64)
    d_intensities = cuda.device_array(total_points, dtype=np.float64)

    d_c_subs = cuda.to_device(c_subs)
    d_c_buff = cuda.to_device(c_buff)
    d_c_A = cuda.to_device(c_A)
    d_c_B = cuda.to_device(c_B)
    d_params_f = cuda.to_device(params_f)
    d_params_i = cuda.to_device(params_i)
    d_params_b = cuda.to_device(params_b)

    stepTheta = angleLimit / divided

    threads_per_block = 256
    blocks_per_grid = (total_points + (threads_per_block - 1)) // threads_per_block

    exec_times = []
    cpu_usages = []
    memory_usages = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(exec_times, cpu_usages, memory_usages, stop_event))
    monitor_thread.start()

    print(f"Launching CUDA Kernel: Blocks={blocks_per_grid}, Threads={threads_per_block}")
    start_gpu = time.time()

    calculate_spectrum_kernel[blocks_per_grid, threads_per_block](
        minTheta, stepTheta, select_version_id,
        d_c_subs, d_c_buff, d_c_A, d_c_B,
        d_params_f, d_params_i, d_params_b,
        d_angles, d_intensities
    )
    cuda.synchronize()

    end_gpu = time.time()
    print(f"GPU Calculation time: {end_gpu - start_gpu:.4f} s")

    angles = d_angles.copy_to_host()
    intensities = d_intensities.copy_to_host()

    stop_event.set()
    monitor_thread.join()

    fileName = f"res_lsmo_cuda_th_{extMinTh}_{select_version}_{extMaxTh}.txt"
    with open(fileName, "w") as streamWriter:
        for i, val in enumerate(intensities):
            if angles[i] >= extMinTh and angles[i] < extMaxTh:
                streamWriter.write(f"{angles[i]:20.10e}{val / divider:20.10e}\n")

    return angles, intensities, exec_times, cpu_usages, memory_usages, divided, extMinTh, extMaxTh, dA, dB, select_version

def analyze_diffraction(func, kwargs):
    versions = kwargs['select_version'].split()
    ordered_versions = [v for v in ['ideal1', 'ideal2', 'monte_carlo'] if v in versions]

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

        ax1.semilogy(filtered_angles, filtered_intensities, color=colors[i], linewidth=1.5, label=f'{version} (x{multipliers[i]})')

    ax1.set_xlabel("2θ [°]", fontsize=12)
    ax1.set_ylabel("Intensity [arb. units]", fontsize=12)
    ax1.set_title("Diffraction Pattern Comparison (CUDA Accelerated)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2 = plt.subplot(gs[-3, 0])
    ax2.plot(exec_times, color="blue")
    ax2.set_title("Total Execution Time")

    ax3 = plt.subplot(gs[-3, 1])
    ax3.plot(cpu_usages, color="red")
    ax3.set_title("Host CPU Usage")

    ax4 = plt.subplot(gs[-3, 2])
    ax4.plot(memory_usages, color="green")
    ax4.set_title("Host RAM Usage")

    plt.tight_layout()
    plt.show()

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate diffraction patterns (CUDA Version)')
    parser.add_argument('--divided', type=int, default=3600)
    parser.add_argument('--minTheta', type=float, default=0.0)
    parser.add_argument('--angleLimit', type=float, default=90.0)
    parser.add_argument('--extMinTh', type=float, default=30.0)
    parser.add_argument('--extMaxTh', type=float, default=60.0)
    parser.add_argument('--dA', type=float, default=1.938)
    parser.add_argument('--dB', type=float, default=2.018)
    parser.add_argument('--divider', type=int, default=1)
    parser.add_argument('--select_version', type=str, required=True)
    parser.add_argument('--dSubs', type=float, default=1.9525)
    parser.add_argument('--dBuff', type=float, default=1.98)
    parser.add_argument('--nSubs', type=int, default=20000)
    parser.add_argument('--nBuff', type=int, default=1)
    parser.add_argument('--nA', type=int, default=14)
    parser.add_argument('--mB', type=int, default=9)
    parser.add_argument('--nRepeat', type=int, default=6)
    parser.add_argument('--buffer_enabled', type=str2bool, default=True)
    parser.add_argument('--LSMO', type=str2bool, default=True)
    parser.add_argument('--fSubs1', type=str, default='Sr')
    parser.add_argument('--fSubs2', type=str, default='Ti')
    parser.add_argument('--fSubs3', type=str, default='O')
    parser.add_argument('--fBuff1', type=str, default='Sr')
    parser.add_argument('--fBuff2', type=str, default='Ti')
    parser.add_argument('--fBuff3', type=str, default='O')
    parser.add_argument('--fA1a', type=str, default='La')
    parser.add_argument('--fA1b', type=str, default='Sr')
    parser.add_argument('--fA2', type=str, default='Mn')
    parser.add_argument('--fA3', type=str, default='O')
    parser.add_argument('--fB1', type=str, default='Ba')
    parser.add_argument('--fB2', type=str, default='Ti')
    parser.add_argument('--fB3', type=str, default='O')
    return parser.parse_args()

def main():
    if not cuda.is_available():
        print("Error: CUDA is not available on this system.")
        sys.exit(1)

    args = parse_arguments()
    kwargs = vars(args)

    print("Analyzing intensity (CUDA GPU) with parameters:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

    print("\nStarting CUDA calculation...")
    analyze_diffraction(geo_intensity_cuda, kwargs)

if __name__ == "__main__":
    main()