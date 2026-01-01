# ECE408 – Parallel Programming with CUDA

This repository contains CUDA-based implementations and experiments developed as part of **ECE408: Parallel Programming**.  
The work focuses on **GPU kernel design, memory hierarchy awareness, and performance-critical parallel algorithms**.

The notebooks intentionally retain **executed outputs**, which document correctness checks, intermediate results, and observed performance behavior during experimentation.

---

## 📂 Repository Contents

### 1. Core ECE408 Assignments  
**`ECE408_Assignments.ipynb`**

This notebook contains CUDA implementations of foundational and advanced parallel programming problems, emphasizing algorithm–architecture interaction.

Covered topics include:

- **Vector Addition**
  - Baseline GPU kernel
  - Thread indexing and grid–block mapping
  - Comparison with CPU execution

- **Sum Reduction**
  - Parallel reduction patterns
  - Shared-memory optimization
  - Synchronization and bank-conflict considerations

- **Brent–Kung Parallel Scan**
  - Work-efficient scan algorithm
  - Reduction and downsweep phases
  - Trade-offs between parallelism and synchronization

- **Matrix Multiplication**
  - Naive GPU matrix multiplication
  - Global memory access patterns

- **Tiled Matrix Multiplication**
  - Shared-memory tiling strategy
  - Improved memory reuse and coalescing
  - Performance comparison with naive implementation

- **3D Convolution**
  - CUDA kernels for volumetric convolution
  - Shared-memory reuse for stencil operations
  - Boundary handling and correctness verification

- **Sparse Data Representation (JDS Format)**
  - Jagged Diagonal Storage (JDS) for sparse matrices
  - Reordering for improved memory coalescing
  - GPU-friendly sparse computation strategies

- **GPU Information and Device Query**
  - CUDA device properties
  - Thread/block limits
  - Shared memory and register availability
  - Relating hardware constraints to kernel design choices

The outputs included in the notebook demonstrate:
- Correctness validation
- Intermediate debugging results
- Observed runtime and scaling behavior on GPU

---

### 2. Histogramming and Image Equalization  
**`Histogram_Equalization_CUDA.ipynb`**

This notebook implements a GPU-based histogramming and histogram equalization pipeline, a canonical example of a **memory- and contention-bound workload**.

Key aspects explored:
- Parallel histogram construction
- Atomic operation contention
- Shared-memory privatization strategies
- Trade-offs between synchronization overhead and memory traffic

Outputs are retained to show:
- Histogram correctness
- Image transformation results
- Kernel behavior across stages of the algorithm

---

## 🧠 Concepts and Skills Demonstrated

- CUDA programming model (threads, blocks, grids)
- Memory hierarchy awareness (global vs shared memory)
- Parallel primitives:
  - Vector addition
  - Reduction
  - Scan (Brent–Kung)
- Dense linear algebra on GPU (matrix multiplication)
- Shared-memory tiling techniques
- Stencil-based computation (3D convolution)
- Sparse data layouts and GPU-friendly formats (JDS)
- GPU hardware introspection and resource-aware design
- Performance bottlenecks and trade-offs in GPU systems

---

## 🛠️ Tools and Environment

- **CUDA C/C++**
- **Google Colab** (GPU runtime)
- **Python** for validation and visualization

Notebooks are designed to be self-contained and executable in a GPU-enabled Colab environment.

---

## 📊 Notes on Outputs and Performance

The presence of outputs in the notebooks is intentional.  
They document:
- Validation steps
- Algorithm behavior
- Performance-relevant observations

The implementations prioritize **clarity, correctness, and architectural understanding** rather than aggressive micro-optimizations.  
Potential extensions include:
- Warp-level primitives
- Further reduction of atomic contention
- Profiling-guided tuning using NVIDIA Nsight

---

## 🔬 Research-Oriented Observations

- Reduction and scan kernels highlight synchronization and shared-memory trade-offs.
- Tiled matrix multiplication demonstrates the impact of memory reuse on arithmetic intensity.
- 3D convolution performance is constrained by shared-memory capacity and access patterns.
- Sparse formats such as JDS expose irregular memory behavior and coalescing challenges.
- These experiments reinforce the importance of **algorithm–architecture co-design** in GPU systems.

---

## ⚠️ Academic Integrity Notice

This repository is intended for **learning and reference purposes only**.  
Do **not** submit this code as part of any graded coursework.

---

## 📜 License

MIT License
