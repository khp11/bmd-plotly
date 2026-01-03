#TASK 1
'''
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("screening_task.nc")   # or the correct name
df1 = ds['forces'].to_pandas()
df1

"""TASK 1:Create a shear force (SFD) and bending moment diagram (BMD) for the central longitudinal girder (central line) consisting of the elements [15,24,33,42,51,60,69,78,83]."""

with open("element.py" ,"r+") as f:
  dictdata= eval("{"+" ".join(f.readlines()[2:]))
df=pd.DataFrame({"member_id" : dictdata.keys(),
"start_node_id" : [i[0] for i in dictdata.values()],
 "end_node_id": [i[-1] for i in dictdata.values()]})

"""#from image we conclude and it also given that : central grider member ids are : [15,24,33,42,51,60,69,78,83]"""

#extracting data of only central grider for task 1

dft1= df[df["member_id"].isin([15,24,33,42,51,60,69,78,83])]

dft1

df1.columns = [i.strip() for  i in df1.columns]
#removing extraspace as its common to have in columns:-)

df1

l= [15,24,33,42,51,60,69,78,83]
rawfinal=df1.loc[l]

#for bending moment diagram

rawbmd=rawfinal[['Mz_i',"Mz_j"]]

xaxisnode=np.append(dft1["start_node_id"].values,dft1["end_node_id"].values[-1])

xaxisnode

#using consistent bending moment directin as each member have its own local codinates axis
#in my case i took positive

yaxis=np.append(rawbmd["Mz_i"].values,rawbmd["Mz_j"].values[-1])
yaxisf= np.array(list(map(float,yaxis)))

convertedy = list(map(float,[format(x, 'f') for x in yaxisf]))

with open("node.py","r+") as fn:
  xcordinates=pd.DataFrame(eval("{"+ "".join(fn.readlines()[2:])))

xcord=xcordinates[xaxisnode].loc[0].values

#naming the data correctly
nodes= xaxisnode
X = xcord
Y= convertedy

plt.figure(figsize=(14,8))

# ---------------------------------------------------------
# Plot the BMD line
# ---------------------------------------------------------
plt.plot(X, Y, linewidth=2.5, color="green")

# Node markers
plt.scatter(X, Y, color="red", zorder=5)

# ---------------------------------------------------------
# Label each node number + moment value
# ---------------------------------------------------------
for i in range(len(X)):
    # Node number
    plt.text(X[i],
             Y[i] - (max(Y)*0.05),    # slight below point
             f"N{nodes[i]}",
             ha='center', fontsize=10, color="blue")

    # Moment value
    plt.text(X[i],
             Y[i] + (max(Y)*0.02),
             f"M={Y[i]:.2f}",
             ha='center', fontsize=9, color="red" )

# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------
plt.title("Bending Moment Diagram (BMD) – Central Longitudinal Girder", fontsize=16)
plt.xlabel("Bridge Length (m)", fontsize=13)
plt.ylabel("Bending Moment Mz (kN·m)", fontsize=13)

plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

plt.xticks(X)
plt.yticks(np.linspace(min(Y), max(Y), 24))

plt.tight_layout()

plt.axvline(11.11, linestyle='--')
plt.text(11.11, 0.3, "Max at Node 28 with bridge length :11.11m", rotation=90 , color = "red",fontsize =15)
plt.savefig("task1-bmd.png", dpi=300)
plt.show()

rawssd= rawfinal[["Vy_i", "Vy_j"]]

#taking global convention for whole
rawssd.loc[:,"Vy_j"]=rawssd["Vy_i"]
Vyfinal = rawssd["Vy_i"].values

x_nodes =X
Vy_single=Vyfinal
Vy_plot = np.repeat(Vyfinal, 2)          # repeat each shear value twice
x_plot  = np.repeat(X, 2)[1:-1]            # repeat each x except ends


plt.figure(figsize=(14,8))

plt.step(x_plot, Vy_plot, where='post', linewidth=2, color='blue')

# zero line
plt.axhline(0, color='black', linewidth=1)

# node markers
plt.scatter(x_nodes, np.zeros_like(x_nodes), color='red', zorder=5)

# label the nodes
for i, x in enumerate(x_nodes):
    plt.text(x, 0.15, f"N{nodes[i]}", ha='center', fontsize=9, color='red')

# shear values annotation
for i in range(len(Vy_single)):
    mid = (x_nodes[i] + x_nodes[i+1]) / 2

    if (i==2):

      plt.text(mid, Vy_single[i] + 0.12*np.sign(Vy_single[i]),
              f"Vy(kN)= {Vy_single[i]:.2f} Minimum value",
              ha='center', fontsize=11, color='red')
    elif (i==4):
      plt.text(mid, Vy_single[i] + 0.12*np.sign(Vy_single[i]),
              f"Vy(kN)= {Vy_single[i]:.2f} Maximum value",
              ha='center', fontsize=11, color='red')

    else:
      plt.text(mid, Vy_single[i] + 0.12*np.sign(Vy_single[i]),
              f"Vy(kN)= {Vy_single[i]:.2f}",
              ha='center', fontsize=11, color='green')

# formatting
plt.title("Shear Force Diagram (SFD) – Central Longitudinal Girder", fontsize=16)
plt.xlabel("Bridge Length (m)", fontsize=13)
plt.ylabel("Shear Force Vy (kN)", fontsize=13)

plt.xticks(x_nodes)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


plt.tight_layout()

plt.savefig("task1-sfd.png", dpi=300)


plt.show()
'''
#TASK 2

import xarray as xr
import numpy as np
import plotly.graph_objects as go

# LOAD NODES & ELEMENT CONNECTIVITY

nodes = {}
elements = {}
exec(open("node.py").read(), nodes)
exec(open("element.py").read(), elements)

nodes = nodes["nodes"]
members = elements["members"]

# LOAD INTERNAL FORCES (NETCDF)
ds = xr.open_dataset("screening_task.nc")

def find_component(name):
    for c in ds["Component"].values:
        if c.lower() == name.lower():
            return c
    return None

Vy_i = find_component("Vy_i")
Vy_j = find_component("Vy_j")
Mz_i = find_component("Mz_i")
Mz_j = find_component("Mz_j")

def get_force(elem, comp):
    return float(ds["forces"].sel(Element=elem, Component=comp).values)

# GIRDER GROUPING
girders = {
    5: list(range(13, 77, 9))+[81],
    4: list(range(14, 78, 9))+[82],
    3: list(range(15, 79, 9))+[83],
    2: list(range(16, 80, 9))+[84],
    1: list(range(17, 81, 9)) +[85],

    
    
    
    
}
'''
colors = {
    1: "red",
    2: "orange",
    3: "green",
    4: "blue",
    5: "purple"
}
'''
# BUILD GIRDER POLYLINES
def build_polyline(elem_list, comp_i, comp_j):
    xs, ys, zs, vals, node_ids = [], [], [], [], []

    for e in elem_list:
        n1, n2 = members[e]
        x1, y1, z1 = nodes[n1]

        xs.append(x1)
        ys.append(y1)
        zs.append(z1)
        vals.append(get_force(e, comp_i))
        node_ids.append(n1)

    # Last end node
    last_e = elem_list[-1]
    n1, n2 = members[last_e]
    x2, y2, z2 = nodes[n2]

    xs.append(x2)
    ys.append(y2)
    zs.append(z2)
    vals.append(get_force(last_e, comp_j))
    node_ids.append(n2)

    return np.array(xs), np.array(ys), np.array(zs), np.array(vals), node_ids
'''
# PLOTLY 3D SFD INTERACTIVE
fig_sfd = go.Figure()

for gid, elems in girders.items():
    xs, ys, zs, vy, node_ids = build_polyline(elems, Vy_i, Vy_j)
    y_plot = ys + vy * 0.2  # shear scale

    hover_text = [
        f"Node {nid}<br>X = {x:.3f}<br>Vy = {v:.3f}"
        for nid, x, v in zip(node_ids, xs, vy)
    ]

    fig_sfd.add_trace(go.Scatter3d(
        x=xs,
        y=y_plot,
        z=zs,
        mode='lines+markers',
        line=dict(color=colors[gid], width=6),
        marker=dict(size=4, color=colors[gid]),
        name=f"Girder {gid}",
        text=hover_text,
        hoverinfo="text"
    ))

fig_sfd.update_layout(
    title="Interactive 3D SFD (Shear Force Diagram):hover on nodes for more info",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Shear Shifted Y",
        zaxis_title="Z",
    ),
    legend=dict(title="Girders")
)
fig_sfd.write_html("sfd_interactive.html")

fig_sfd.show()
'''

# PLOTLY 3D BMD INTERACTIVE
fig_bmd = go.Figure()
xfull=[]
mzfull =[]
for gid, elems in girders.items():
    xs, ys, zs, mz, node_ids = build_polyline(elems, Mz_i, Mz_j)
    mz=100*mz
    y_plot = mz #* 0.05  # moment scale
    xfull.extend(xs)
    mzfull.extend(mz)
    hover_text = [
        f"Node {nid}<br>X = {x:.3f}<br>Mz = {v:.3f}<br>Z = {z:.3f}"
        for nid, x, v,z in zip(node_ids, xs, mz,zs)
    ]

    fig_bmd.add_trace(go.Scatter3d(
        x=xs,
        y=y_plot,
        z=zs,
        mode='lines+markers',
        line=dict(color="red", width=4),
        marker=dict(size=3, color="red"),
        #name=f"Girder {gid}",
        showlegend=False,
        text=hover_text,
        hoverinfo="text"
    ))
    

    #baseline
    fig_bmd.add_trace(go.Scatter3d(
    x=[xs[0], xs[-1]],     # first and last node X
    y=[0, 0],             # Mz = 0 baseline
    z=[zs[0], zs[0]],    # same girder elevation
    mode='lines',
    line=dict(
        color="green",
        width=3,
        dash='solid'
    ),
    showlegend=False,
    hoverinfo='skip'
))  
  
    fig_bmd.add_trace(go.Scatter3d(
    x = [xs[np.argmax(mz)],xs[np.argmax(mz)]],
    y = [0, mz[np.argmax(mz)]],
    z = [zs[0]] * 2,
    mode="lines",
    line=dict(color="black", width=3),
    legendgroup="max_lines",
    showlegend=False,
    visible=False,          #  start OFF
    hoverinfo="skip"
))

    fig_bmd.add_trace(go.Scatter3d(
        x = [xs[np.argmin(mz)]] * 2,
        y = [0, mz[np.argmin(mz)]],
        z = [zs[0]] * 2,
        mode="lines",
        line=dict(color="black", width=3),
        legendgroup="min_lines",
        showlegend=False,
        visible=False,          #  start OFF
        hoverinfo="skip"
    ))


    # ================= SUPPORT GEOMETRY =================
    L = max(xs) - min(xs)
    h = 0.1299 * L      # support height
    w = 0.03 * L       # support half-width
    r = 0.01 * L       # roller radius

    # ---------- PIN SUPPORT (START NODE) ----------
    x0, z0 = xs[0], zs[0]

    fig_bmd.add_trace(go.Scatter3d(
        x=[x0 - w, x0, x0 + w, x0 - w],
        y=[-h, 0, -h, -h],
        z=[z0, z0, z0, z0],
        mode="lines",
        line=dict(color="blue", width=5),
        showlegend=False,
        hoverinfo="skip"
    ))

    # --------ROLLER SUPPORT(END NODE)--------------
    x1, z1 = xs[-1], zs[-1]

    # Triangle
    fig_bmd.add_trace(go.Scatter3d(
        x=[x1 - w, x1, x1 + w, x1 - w],
        y=[-h, 0, -h, -h],
        z=[z1, z1, z1, z1],
        mode="lines",
        line=dict(color="blue", width=5),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Rollers (two wheels)
    theta = np.linspace(0, 2 * np.pi, 40)

    for dx in [-w / 2, w / 2]:
        fig_bmd.add_trace(go.Scatter3d(
            x=x1 + dx + r * np.cos(theta),
            y=-h - r + (r*5/2) * np.sin(theta),
            z=[z1] * len(theta),
            mode="lines",
            line=dict(color="blue", width=5),
            showlegend=False,
            hoverinfo="skip"
        ))

pad = 0.01*(max(xfull) - min(xfull))
#pad2 = 0.01*(max(mzfull) - min(mzfull))


max_idx = [i for i, t in enumerate(fig_bmd.data) if t.legendgroup == "max_lines"]
min_idx = [i for i, t in enumerate(fig_bmd.data) if t.legendgroup == "min_lines"]


fig_bmd.update_layout(
updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=1.15,
            showactive=True,
            active=-1,
            buttons=[
                # -------- MAX TOGGLE --------
                dict(
                    label="MAX",
                    method="update",

                    # OFF → ON
                    args=[{
                        "visible": [
                            True if t.legendgroup == "max_lines" else t.visible
                            for t in fig_bmd.data
                        ]
                    }],

                    # ON → OFF
                    args2=[{
                        "visible": [
                            False if t.legendgroup == "max_lines" else t.visible
                            for t in fig_bmd.data
                        ]
                    }]
                ),

                # -------- MIN TOGGLE --------
                dict(
                    label="MIN",
                    method="update",

                    # OFF → ON
                    args=[{
                        "visible": [
                            True if t.legendgroup == "min_lines" else t.visible
                            for t in fig_bmd.data
                        ]
                    }],

                    # ON → OFF
                    args2=[{
                        "visible": [
                            False if t.legendgroup == "min_lines" else t.visible
                            for t in fig_bmd.data
                        ]
                    }]
                ),
            ]
        )
    ],  
    title="Interactive 3D BMD",
    scene=dict(
        
        camera=dict(
            eye=dict(x=0.001, y=0.001, z=10),  # 👈 looking along +Z
            up=dict(x=0, y=1.5, z=0),           # 👈 Y is vertical (Mz)
            center=dict(x=0, y=0, z=0)
        ),
        xaxis=dict(
            title = "girder length",
            #range=[min(xfull) - pad, max(xfull) + pad],
            showbackground=False,   # removes YZ plane
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        yaxis=dict(
            range=[min(mzfull) - 0.2 * (max(xfull) - min(xfull)),max(mzfull) + 0.2 * (max(xfull) - min(xfull))],
            title = "Mz values",
            #range = [min(mzfull)-pad2,max(mzfull)+pad2],
            showbackground=True,    # 
            backgroundcolor="rgba(200,200,200,0.15)",
            showgrid=False,
            zeroline=False,
            visible=False           # showaxis line, keep plane
        ),
        zaxis=dict(
            showbackground=False,   # removes XY plane
            showgrid=False,
            zeroline=False,
            visible=False,
            autorange="reversed"
        ),
        aspectmode='manual',
        aspectratio=dict(
            x=10,   #makes cuboid long in X
            y=2,
            z=5
        )
    ),
    
    
    #legend=dict(title="Girders")
)

fig_bmd.write_html("bmd_interactive.html",include_plotlyjs=True,full_html=True)


fig_bmd.show()