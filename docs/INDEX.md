# 🌙 Lunar Horizon Optimizer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-243%2F243%20passing-brightgreen.svg)](tests/)
[![Pipeline](https://img.shields.io/badge/pipeline-clean-brightgreen.svg)](Makefile)

An integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions.

## 🚀 Interactive 3D Landing Visualizations

**[🌐 View Live 3D Demos →](demos/)**

Experience realistic lunar landing trajectories with:
- 🔥 **Rocket exhaust plumes** showing deceleration phases
- 📐 **Optimized camera positioning** for natural above-surface views
- 🎮 **Interactive controls** with full 3D rotation and zoom
- 📊 **Physics accuracy** with gentle descent angles and soft landings

### Featured Scenarios

| Demo | Engine | Mission | Technology |
|------|--------|---------|------------|
| **[Blue Origin Cargo Express](demos/blue_origin_landing_3d.html)** | BE-7 (CH4/O2) | Commercial cargo delivery | Reusable lander |
| **[Artemis Cargo Lander](demos/artemis_cargo_landing_3d.html)** | RL-10 (LOX/LH2) | NASA south pole mission | Precision landing |
| **[Quick Test Landing](demos/quick_test_landing_3d.html)** | Aestus (hypergolic) | Validation scenario | Fast execution |

## ✨ Key Features

- 🔄 **Differentiable Optimization**: JAX-based gradient optimization
- 💰 **Economic Analysis**: NPV, IRR, ROI with ISRU modeling
- ⚡ **Global Optimization**: Multi-objective PyGMO NSGA-II
- 🛸 **Trajectory Generation**: Lambert solvers, powered descent
- 📈 **3D Visualization**: Enhanced landing trajectory visualization
- 🌍 **Environmental Costs**: Wright's law learning curves

## 🛠️ Generate Your Own

```bash
# Install and setup
git clone https://github.com/ethene/lunar-horizon-optimizer.git
cd lunar-horizon-optimizer
conda create -n py312 python=3.12 -y
conda activate py312
conda install -c conda-forge pykep pygmo astropy spiceypy -y
pip install -r requirements.txt

# Generate 3D landing visualization
./lunar_opt.py run scenario 12_powered_descent_mission --include-descent --3d-viz
```

## 📊 Validation Results

**All scenarios demonstrate realistic physics:**
- ✅ Delta-V: 15.6-16.2 km/s (Earth-Moon transfer range)
- ✅ Descent angles: 26.6° (gentle, < 45° requirement)
- ✅ Landing velocity: 1.0 m/s (< 3 m/s for soft landing)
- ✅ Multi-objective optimization with 1-3% improvements

## 🔗 Links

- **[📖 User Guide](guides/NEW_CLI_USER_GUIDE.md)**
- **[🧪 Test Results](../tests/)**
- **[📋 Scenarios](../scenarios/)**
- **[🔧 API Documentation](../src/)**

---

**Repository**: [github.com/ethene/lunar-horizon-optimizer](https://github.com/ethene/lunar-horizon-optimizer)