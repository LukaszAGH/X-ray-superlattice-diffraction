# ⚡ XRD Spectra Generator (CUDA GPU Edition)

This is a high-performance, GPU-accelerated version of the X-ray diffraction simulator. It utilizes **NVIDIA CUDA** cores via the **Numba** library to parallelize the intensity calculations across thousands of threads, offering significant speedups for high-resolution spectra and Monte Carlo simulations.

## 📦 Requirements

- Python 3.11+
- NVIDIA GPU with properly installed drivers
- CUDA Toolkit (compatible with your drivers)

### Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure `atomic_data.txt` exists in the parent directory (root of the repo).

## 🚀 Usage

The usage is identical to the CPU version, but ran from this folder:

   ```bash
    cd cuda_gpu
    python x_ray_spectra_gen_cuda.py --extMinTh 20 --extMaxTh 80 --select_version monte_carlo
   ```
## 🔧 How it works
- **Kernel Parallelization**: The main loop over angle steps (divided) is unrolled onto the GPU grid.

- **JIT Compilation**: Python functions are compiled to optimized machine code (@cuda.jit).

- **Memory Management**: Atomic scattering factors are pre-processed on the CPU and transferred to GPU global memory to minimize overhead.

- **Identical Logic**: The physics math and Monte Carlo seeding logic (LCG) are ported 1:1 to ensure results match the CPU version exactly.