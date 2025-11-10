# 🧠 Stokes System Resolution – Rigid & Compliant Tube Flow (FVM)

This project implements a numerical solver for incompressible **Stokes flow** in both **rigid** and **flexible** tubes using a **finite-volume MAC scheme**.  
It also models **fluid–structure interaction** via the **Lambert tube law**, allowing realistic airway deformation under pressure gradients.

---

## 🚀 Features

- ✅ Stokes equations solver (low-Re hydrodynamics)
- ✅ Finite Volume Method (MAC grid)
- ✅ Tube wall deformation using Lambert law (soft airways)
- ✅ Pressure & velocity field visualization
- ✅ Flow rate tracking along the tube
- ✅ Hydraulic resistance computation vs Poiseuille theory
- ✅ Comparison rigid vs compliant tube

---

## 📂 Repository Structure

| File | Description |
|------|------------|
| `adam.py` | Main solver – wall deformation (Lambert model) |
| `casrigidelkhr.py` | Rigid tube version (Poiseuille benchmark) |
| `plots/` | Plots & animations (optional) |

---

## 📊 Output Examples

- Pressure field evolution
- Velocity field (u, v components)
- Tube deformation over x
- Flow resistance vs theoretical Poiseuille value

> 💡 Add your GIFs in `/plots/` (instructions below ⬇️)

---

## 🔧 Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy matplotlib
