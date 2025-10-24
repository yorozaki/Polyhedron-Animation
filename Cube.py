import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- setup ----------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.set_box_aspect([1,1,1])

# ---------- base geometry ----------
cube = np.array([[x,y,z] for x in [0,1] for y in [0,1] for z in [0,1]], float)
cube_edges=[(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
            (4,5),(4,6),(5,7),(6,7)]
tetra = cube[[4,2,1,7]]
tetra_edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
tetra_faces=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
octa = np.array([[0.5,0.5,0],[0.5,0.5,1],[0.5,0,0.5],
                 [0.5,1,0.5],[0,0.5,0.5],[1,0.5,0.5]],float)
octa_edges=[(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),
            (1,5),(2,4),(2,5),(3,4),(3,5)]
octa_faces=[[0,2,4],[0,4,3],[0,3,5],[0,5,2],
            [1,4,2],[1,3,4],[1,5,3],[1,2,5]]
center=np.array([0.5,0.5,0.5])

# ---------- helpers ----------
def ease_in_out(t): return 3*t**2 - 2*t**3
def rescale(p,s): return (p-center)*s + center
def rodrigues(v,a,ang):
    a=a/(np.linalg.norm(a)+1e-12)
    K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    R=np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K@K)
    return v@R.T
def draw_partial_edges(points, edges, progress, color, alpha=1, lw=2, dotted=False):
    style=(0,(2,6)) if dotted else 'solid'
    progress=np.clip(progress,0,1)
    total=len(edges)
    nfull=int(progress*total)
    for (i,j) in edges[:nfull]:
        ax.plot([points[i,0],points[j,0]],
                [points[i,1],points[j,1]],
                [points[i,2],points[j,2]],
                color=color,alpha=alpha,lw=lw,linestyle=style)
    if nfull<total and progress>0:
        i,j=edges[nfull]
        t=progress*total-nfull
        pi,pj=points[i],points[j]
        ax.plot([pi[0],pi[0]+(pj[0]-pi[0])*t],
                [pi[1],pi[1]+(pj[1]-pi[1])*t],
                [pi[2],pi[2]+(pj[2]-pi[2])*t],
                color=color,alpha=alpha,lw=lw,linestyle=style)
def draw_faces(p,f,col,a):
    for fc in f:
        pts=np.array([p[i] for i in fc])
        poly=Poly3DCollection([pts],facecolors=col,edgecolors='none',alpha=a)
        ax.add_collection3d(poly)

# ---------- orient octa ----------
O3,O4=octa[3],octa[4]; T2,T3=tetra[2],tetra[3]
v1=O4-O3; v1/=np.linalg.norm(v1)+1e-12
v2=T2-T3; v2/=np.linalg.norm(v2)+1e-12
axis=np.cross(v1,v2); axis/=np.linalg.norm(axis)+1e-12
angle=np.arccos(np.clip(np.dot(v1,v2),-1,1))
oct_center=octa.mean(0)
R_align=rodrigues(np.eye(3),axis,angle)
octa_rot=(octa-oct_center)@R_align.T+oct_center
target=T3; O3f=octa_rot[3]
octa_final=octa_rot+(target-O3f)

# ---------- persistent offsets ----------
shiftA2 = shiftB2 = shiftA4 = shiftB4 = np.zeros(3)

def update(frame):
    global shiftA2, shiftB2, shiftA4, shiftB4
    ax.cla(); ax.set_facecolor("black"); ax.axis("off")
    ax.set_xlim(-0.2,1.2); ax.set_ylim(-0.2,1.2); ax.set_zlim(-0.2,1.2)
    ax.view_init(elev=30,azim=45+frame*0.6); ax.dist=8.6

    def lift(p,z): return p+np.array([0,0,z])
    scale_base=1-0.25*min(max(frame-180,0)/60,1)
    scale_dup =1-0.10*min(max(frame-240,0)/60,1)
    t5=min(max(frame-310,0)/60,1); t6=min(max(frame-380,0)/60,1); t7=min(max(frame-450,0)/60,1)
    svert1=1-0.25*ease_in_out(t5); svert2=1-0.15*ease_in_out(t6); svert3=1-0.20*ease_in_out(t7)
    scale=scale_base*scale_dup*svert1*svert2*svert3; h2=h3=h4=1.0*scale

    # --- Layer 1 ---
    cubeA=rescale(cube,scale); tetraA=rescale(tetra,scale)
    if 120<=frame<=180: octaA=rescale(octa,scale)
    elif 180<frame<=240:
        t3=ease_in_out((frame-180)/60)
        R2=rodrigues(np.eye(3),axis,angle*t3)
        rot=(octa-oct_center)@R2.T+oct_center
        O3r=rot[3]; octaA=rescale(rot+(target-O3r)*t3,scale)
    elif frame>240: octaA=rescale(octa_final,scale)
    else: octaA=None

    # --- Duplicate B ---
    cubeB=tetraB=None
    if frame>=240:
        s_final=0.75; cubeA_s=rescale(cube,s_final); tetraA_s=rescale(tetra,s_final)
        base=np.array([s_final,0,0]); t4=min((frame-240)/60,1)
        path=base*ease_in_out(t4)
        def remap(P): raw=(P-center)/s_final+center; return rescale(raw,scale)
        cubeB=remap(cubeA_s+path); tetraB=remap(tetraA_s+path)

    # --- Layers 2/3/4 ---
    cubeA2=tetraA2=cubeB2=tetraB2=octaA2=None
    if frame>=310 and cubeB is not None:
        z=ease_in_out(min((frame-310)/60,1))*h2
        cubeA2=lift(cubeA,z); tetraA2=lift(tetraA,z)
        cubeB2=lift(cubeB,z); tetraB2=lift(tetraB,z)
        if octaA is not None: octaA2=lift(octaA,z)
    cubeA3=tetraA3=cubeB3=tetraB3=octaA3=None
    if frame>=380 and cubeA2 is not None:
        z=ease_in_out(t6)*h3
        cubeA3=lift(cubeA2,z); tetraA3=lift(tetraA2,z)
        cubeB3=lift(cubeB2,z); tetraB3=lift(tetraB2,z)
        if octaA2 is not None: octaA3=lift(octaA2,z)
    cubeA4=tetraA4=cubeB4=tetraB4=None
    if frame>=450 and cubeA3 is not None:
        z=ease_in_out(t7)*h4
        cubeA4=lift(cubeA3,z); tetraA4=lift(tetraA3,z)
        cubeB4=lift(cubeB3,z); tetraB4=lift(tetraB3,z)

    # --- compute shift vectors once (after layer 4 done) ---
    if frame==530 and octaA is not None and octaA3 is not None:
        O4_1,O5_1=octaA[4],octaA[5]
        O4_3,O5_3=octaA3[4],octaA3[5]
        T1A2,T0B2=tetraA2[1],tetraB2[0]
        T1A4,T0B4=tetraA4[1],tetraB4[0]
        update.shift_targets = (
            O4_1 - T1A2, O5_1 - T0B2,
            O4_3 - T1A4, O5_3 - T0B4
        )

    # --- apply persistent offsets (530–570) ---
    if hasattr(update,"shift_targets"):
        shiftA2_t,shiftB2_t,shiftA4_t,shiftB4_t = update.shift_targets
        if 530<=frame<=540:
            tmove=ease_in_out((frame-530)/10)
            shiftA2 = shiftA2_t*tmove
            shiftB2 = shiftB2_t*tmove
            shiftA4 = shiftA4_t*tmove
            shiftB4 = shiftB4_t*tmove
        elif frame>570:
            shiftA2,shiftB2,shiftA4,shiftB4 = shiftA2_t,shiftB2_t,shiftA4_t,shiftB4_t

    # apply persistent shifts
    for obj,shiftv in [
        (cubeA2,shiftA2),(tetraA2,shiftA2),(cubeB2,shiftB2),(tetraB2,shiftB2),
        (cubeA4,shiftA4),(tetraA4,shiftA4),(cubeB4,shiftB4),(tetraB4,shiftB4)
    ]:
        if obj is not None: obj += shiftv

    # --- recenter ---
    pts=[p for p in [cubeA,tetraA,octaA,cubeB,tetraB,
                     cubeA2,tetraA2,cubeB2,tetraB2,octaA2,
                     cubeA3,tetraA3,cubeB3,tetraB3,octaA3,
                     cubeA4,tetraA4,cubeB4,tetraB4] if p is not None]
    all_pts=np.vstack(pts); shift=center-all_pts.mean(0)
    def S(p): return p+shift

    # --- dotted/face timings (after shift) ---
    if 570<=frame<=590: tcf=(frame-570)/20; dot=tcf; solid=1-tcf
    elif frame>590: dot=1; solid=0
    else: dot=0; solid=1
    if 590<=frame<=650: rev=max(0,1-(frame-590)/60)
    elif frame>650: rev=0
    else: rev=1
    if 650<=frame<=710: tf=ease_in_out((frame-650)/60); facea=tf
    elif frame>710: facea=1
    else: facea=0

    # --- cubes ---
    def cube_cf(C):
        if C is None: return
        draw_partial_edges(S(C),cube_edges,rev,"white",solid,1.2)
        draw_partial_edges(S(C),cube_edges,rev,"white",dot,1.2,True)
    for C in [cubeA,cubeB,cubeA2,cubeB2,cubeA3,cubeB3,cubeA4,cubeB4]: cube_cf(C)

    # --- growth and faces ---
    if 60<=frame<=120: draw_partial_edges(S(tetraA),tetra_edges,(frame-60)/60,"lime",.95,2)
    elif frame>120: draw_partial_edges(S(tetraA),tetra_edges,1,"lime",.95,2)
    if octaA is not None:
        if 120<=frame<=180: draw_partial_edges(S(octaA),octa_edges,(frame-120)/60,"cyan",.95,2)
        else: draw_partial_edges(S(octaA),octa_edges,1,"cyan",.95,2)
    for T in [tetraB,tetraA2,tetraB2,tetraA3,tetraB3,tetraA4,tetraB4]:
        if T is not None: draw_partial_edges(S(T),tetra_edges,1,"lime",.9,2)
    for O in [octaA2,octaA3]:
        if O is not None: draw_partial_edges(S(O),octa_edges,1,"cyan",.9,2)
    if facea>0:
        for T in [tetraA,tetraB,tetraA2,tetraB2,tetraA3,tetraB3,tetraA4,tetraB4]:
            if T is not None: draw_faces(S(T),tetra_faces,"lime",.35*facea)
        for O in [octaA,octaA2,octaA3]:
            if O is not None: draw_faces(S(O),octa_faces,"cyan",.3*facea)
    return []

ani=FuncAnimation(fig,update,frames=900,interval=50,blit=False)
plt.show()
