### 3D Polyhedron Animation ###

This project visualizes a dynamic transformation sequence between a **cube**, **tetrahedron**, and **octahedron** using **Python**, **NumPy**, and **Matplotlib**.  
The animation demonstrates geometric formation, scaling, and rotation, exploring the aesthetic and mathematical relationships between fundamental 3D solids.

## Concept ##

This project explores geometric morphing and transformation logic as a form of computational art.
Each phase reflects a shift between regular polyhedra, illustrating structural relationships and dynamic balance.
Originally designed as part of an architectural research experiment on parametric motion and responsive geometry.

## Author ##

Yorozaki (Md Rakibul Hasan)
Architect, designer, and researcher exploring intersections between geometry, robotics, and computational design.
<yorozakifahim@gmail.com>

---

## Features

- Real-time 3D animation of cube to tetrahedron to octahedron sequence  
- Smooth, eased transitions using Rodrigues rotation and custom easing functions  
- Layered duplication with synchronized motion and perspective rotation  
- Dark cinematic render style (black background, cyan and lime highlights)  
- Optional export to `.mp4` video via Matplotlib’s `FuncAnimation`

---

## Technologies Used

- **Python 3.13**
- **NumPy** - mathematical computation and vector manipulation  
- **Matplotlib (3D toolkit)** - geometric rendering and animation  
- **FuncAnimation** - for generating frame-by-frame motion  

---

## ⚙️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yorozaki/CubeAnimation.git
   cd CubeAnimation
2. Install dependencies (recommended via Anaconda)

- conda install numpy matplotlib
- or
- pip install numpy matplotlib

3. Run the script

- python Cube.py

4. Save the Video (optional)

- ani.save('CubeAnimation.mp4', writer='ffmpeg', fps=30)

This project is licensed under the MIT License – see the LICENSE file for details.
