import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple

# ====================== 核心数据结构与算法 (保持不变) ======================
@dataclass
class SKU:
    id: str
    l: float; w: float; h: float; qty: int

@dataclass
class BoxType:
    id: str
    l: float; w: float; h: float

@dataclass
class Placement:
    sku_id: str
    pos: Tuple[float,float,float]
    dim: Tuple[float,float,float]
    box_id: str

def orientations(dim):
    l,w,h = dim
    return list(set(itertools.permutations((l,w,h), 3)))

def volume(dim): return dim[0]*dim[1]*dim[2]

def intersect(cube1, cube2):
    x1,y1,z1,l1,w1,h1 = cube1
    x2,y2,z2,l2,w2,h2 = cube2
    return not (x1+l1 <= x2 or x2+l2 <= x1 or y1+w1 <= y2 or y2+w2 <= y1 or z1+h1 <= z2 or z2+h2 <= z1)

def subtract_space(free_spaces, placed):
    new_spaces = []
    px,py,pz,pl,pw,ph = placed
    for fs in free_spaces:
        fx,fy,fz,fl,fw,fh = fs
        if not intersect(fs, placed):
            new_spaces.append(fs); continue
        if fx < px: new_spaces.append((fx, fy, fz, px-fx, fw, fh))
        if px+pl < fx+fl: new_spaces.append((px+pl, fy, fz, fx+fl-(px+pl), fw, fh))
        if fy < py: new_spaces.append((fx, fy, fz, fl, py-fy, fh))
        if py+pw < fy+fw: new_spaces.append((fx, py+pw, fz, fl, fy+fw-(py+pw), fh))
        if fz < pz: new_spaces.append((fx, fy, fz, fl, fw, pz-fz))
        if pz+ph < fz+fh: new_spaces.append((fx, fy, pz+ph, fl, fw, fz+fh-(pz+ph)))
    new_spaces = [s for s in new_spaces if s[3]>1e-9 and s[4]>1e-9 and s[5]>1e-9]
    final_spaces = []
    for i, s1 in enumerate(new_spaces):
        contained = False
        for j, s2 in enumerate(new_spaces):
            if i==j: continue
            if (s2[0]<=s1[0] and s2[1]<=s1[1] and s2[2]<=s1[2] and s1[0]+s1[3]<=s2[0]+s2[3] and s1[1]+s1[4]<=s2[1]+s2[4] and s1[2]+s1[5]<=s2[2]+s2[5]):
                contained = True; break
        if not contained: final_spaces.append(s1)
    return final_spaces

def try_pack_one_box(box, skus):
    free_spaces = [(0.0,0.0,0.0, box.l, box.w, box.h)]
    placements = []
    packed_count = {s.id:0 for s in skus}
    items = [(s.id, (s.l, s.w, s.h)) for s in skus for _ in range(s.qty)]
    for sku_id, dim in items:
        best_choice = None; best_score = None
        for fi, fs in enumerate(free_spaces):
            for ori in orientations(dim):
                if ori[0] <= fs[3]+1e-9 and ori[1] <= fs[4]+1e-9 and ori[2] <= fs[5]+1e-9:
                    score = (fs[3]-ori[0])*fs[4]*fs[5] + fs[3]*(fs[4]-ori[1])*fs[5] + fs[3]*fs[4]*(fs[5]-ori[2])
                    if best_score is None or score < best_score:
                        best_score = score; best_choice = (fi, ori, (fs[0],fs[1],fs[2]))
        if best_choice:
            fi, ori, origin = best_choice
            placements.append(Placement(sku_id, origin, ori, box.id))
            packed_count[sku_id] += 1
            free_spaces = subtract_space(free_spaces, (*origin, *ori))
    return placements, packed_count

def pack_across_boxes(boxes, skus):
    results = []
    total_qty = sum(s.qty for s in skus)
    for b in boxes:
        placements, packed_count = try_pack_one_box(b, skus)
        packed_total = sum(packed_count.values())
        util = sum(volume(p.dim) for p in placements) / volume((b.l, b.w, b.h))
        results.append({"box_id": b.id, "box_dims": (b.l, b.w, b.h), "packed_total": packed_total, "utilization": util, "fits_all": packed_total == total_qty, "placements": placements})
    
    max_box_vol = max(volume((b.l, b.w, b.h)) for b in boxes)
    fit_candidates = [r for r in results if r["fits_all"]]
    if fit_candidates:
        best = max(fit_candidates, key=lambda r: r["utilization"])
    else:
        best = max(results, key=lambda r: r["packed_total"])
        best["forced_max_box"] = True
    return best, results



def draw_placements_interactive(box: BoxType, placements: List[Placement]):
    """
    使用 Plotly 交互式显示装箱结果
    每个 SKU 用不同颜色显示 + 边界线 + 序号标记
    """
    fig = go.Figure()

    # === 绘制外箱框架 ===
    vertices_box = np.array([
        [0, 0, 0],
        [box.l, 0, 0],
        [box.l, box.w, 0],
        [0, box.w, 0],
        [0, 0, box.h],
        [box.l, 0, box.h],
        [box.l, box.w, box.h],
        [0, box.w, box.h]
    ])
    edges_box = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
    for e in edges_box:
        fig.add_trace(go.Scatter3d(
            x=[vertices_box[e[0],0], vertices_box[e[1],0]],
            y=[vertices_box[e[0],1], vertices_box[e[1],1]],
            z=[vertices_box[e[0],2], vertices_box[e[1],2]],
            mode="lines",
            line=dict(color="lightgrey", width=4),
            showlegend=False
        ))

    # === 绘制 SKU 长方体 ===
    random.seed(42)
    for idx, p in enumerate(placements):
        cx, cy, cz = p.pos
        cl, cw, ch = p.dim
        color = f"rgb({random.randint(50,255)},{random.randint(50,255)},{random.randint(50,255)})"

        # 顶点
        vertices = np.array([
            [cx,     cy,     cz],
            [cx+cl,  cy,     cz],
            [cx+cl,  cy+cw,  cz],
            [cx,     cy+cw,  cz],
            [cx,     cy,     cz+ch],
            [cx+cl,  cy,     cz+ch],
            [cx+cl,  cy+cw,  cz+ch],
            [cx,     cy+cw,  cz+ch]
        ])
        faces = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4],
            [2,3,7,6], [1,2,6,5], [0,3,7,4]
        ]
        triangles = []
        for f in faces:
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
        i, j, k = zip(*triangles)

        fig.add_trace(go.Mesh3d(
            x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=i, j=j, k=k,
            color=color, opacity=0.6,
            name=f"SKU-{idx+1}",
            hovertext=f"SKU-{idx+1}<br>pos:{p.pos}<br>dim:{p.dim}",
            hoverinfo="text"
        ))

        # 边界线
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            fig.add_trace(go.Scatter3d(
                x=[vertices[e[0],0], vertices[e[1],0]],
                y=[vertices[e[0],1], vertices[e[1],1]],
                z=[vertices[e[0],2], vertices[e[1],2]],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=False
            ))

        # 序号
        center = vertices.mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="text",
            text=[str(idx+1)],
            textfont=dict(color="black", size=18),
            showlegend=False
        ))

        # SKU 尺寸
        fig.add_trace(go.Scatter3d(
            x=[center[0] + cl*0.6],
            y=[center[1] + cw*0.6],
            z=[center[2] + ch*0.6],
            mode="text",
            text=[f"({cl}, {cw}, {ch})"],
            textfont=dict(color="blue", size=14),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="长度 (cm)"),
            yaxis=dict(title="宽度 (cm)"),
            zaxis=dict(title="高度 (cm)"),
            aspectmode="data"
        ),
        title=f"{box.id} - 交互式装箱可视化"
    )

    fig.add_trace(go.Scatter3d(
        x=[box.l/2], 
        y=[box.w/2], 
        z=[box.h + box.h*0.05],
        mode="text",
        text=[f"{box.id} ({box.l}, {box.w}, {box.h})"],
        textfont=dict(color="red", size=16),
        showlegend=False
    ))

    return fig


# ====================== Streamlit 绘图逻辑 ======================
# def draw_plotly_st(box, placements):
#     fig = go.Figure()
#     # 绘制外箱
#     l, w, h = box.l, box.w, box.h
#     v = np.array([[0,0,0],[l,0,0],[l,w,0],[0,w,0],[0,0,h],[l,0,h],[l,w,h],[0,w,h]])
#     edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
#     for e in edges:
#         fig.add_trace(go.Scatter3d(x=[v[e[0],0], v[e[1],0]], y=[v[e[0],1], v[e[1],1]], z=[v[e[0],2], v[e[1],2]], mode='lines', line=dict(color='black', width=2), showlegend=False))
    
#     # 绘制物品
#     for i, p in enumerate(placements):
#         x, y, z = p.pos
#         dl, dw, dh = p.dim
#         color = f"rgb({random.randint(50,200)},{random.randint(50,200)},{random.randint(50,200)})"
#         fig.add_trace(go.Mesh3d(x=[x,x+dl,x+dl,x,x,x+dl,x+dl,x], y=[y,y,y+dw,y+dw,y,y,y+dw,y+dw], z=[z,z,z,z,z+dh,z+dh,z+dh,z+dh], i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color=color, opacity=0.7, name=f"SKU {i+1}"))
    
#     fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
#     return fig

# ====================== Streamlit 界面 ======================
st.set_page_config(page_title="装箱计算器", layout="wide")
st.title("📦 3D 智能装箱助手")

# 3d装箱用-箱子行政规格长宽+0.5cm，底边不变
BOX_SPECS = {
    "饰品1号箱": (17.5, 20.5, 3.5),
    "饰品2号箱": (25.5, 25.0, 10.0),
    "小1号箱": (27.5, 24.5, 20.0),
    "1号箱": (30.5, 28.5, 22.0),
    "2号箱": (30.5, 27.5, 28.0),
    "16号箱2+1": (34.5, 28.5, 30.0),
    "3号箱": (38.0, 30.0, 32.0),
    "17号箱3+1": (39.5, 32.5, 32.0),
    "4号箱": (41.5, 35.5, 28.5),
    "18号箱4+1": (43.5, 38.5, 32.0),
    "14号箱": (45.5, 40.5, 35.0),
    "14+3": (54.0, 49.0, 27.0),
    "14+1": (55.5, 50.5, 28.5),
    "5号箱": (55.5, 50.5, 30.0),
    "5+1": (57.5, 51.5, 32.0)
}

BOXES = [BoxType(name, *dims) for name, dims in BOX_SPECS.items()]

with st.sidebar:
    st.header("输入物品参数 (mm)")
    in_l = st.number_input("长度 (mm)", value=277.0)
    in_w = st.number_input("宽度 (mm)", value=198.0)
    in_h = st.number_input("高度 (mm)", value=90.0)
    in_q = st.number_input("数量", value=12, min_value=1)
    btn = st.button("开始寻找最佳箱型", type="primary")

# 1. 初始化 Session State（如果不存在）
if 'best_res' not in st.session_state:
    st.session_state.best_res = None
if 'all_res' not in st.session_state:
    st.session_state.all_res = None

# 2. 当点击“寻找箱型”时，将结果存入 session_state
if btn:
    l_cm, w_cm, h_cm = math.ceil(in_l/10), math.ceil(in_w/10), math.ceil(in_h/10)
    best, all_res = pack_across_boxes(BOXES, [SKU("Item", l_cm, w_cm, h_cm, int(in_q))])
    st.session_state.best_res = best
    st.session_state.all_res = all_res

# 3. 只要 session_state 里有结果，就显示界面
if st.session_state.best_res:
    best = st.session_state.best_res
    all_res = st.session_state.all_res
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.success(f"### 推荐：{best['box_id']}&nbsp;&nbsp;&nbsp;&nbsp;箱规格cm：{best['box_dims']}")
        st.metric("空间利用率", f"{best['utilization']:.1%}")
        st.write(f"**已装入：** {best['packed_total']} / {int(in_q)} 件")
        
        with st.expander("查看所有箱型对比"):
            st.table([{"箱型": r["box_id"], "件数": r["packed_total"], "利用率": f"{r['utilization']:.1%}"} for r in all_res])
    
    with c2:
        bx = next(b for b in BOXES if b.id == best["box_id"])
        st.subheader("📦 装箱方案预览")
        
        # 这里的按钮点击后页面会刷新，但因为数据在 session_state 里，所以不会消失
        if st.button("生成 3D 可视化图表 →", type="secondary"):
            with st.spinner("正在绘制 3D 模型..."):
                fig = draw_placements_interactive(bx, best["placements"])
                fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig, use_container_width=True, theme=None)
                st.caption("📱 手机提示：单指旋转，双指捏合缩放，双指拖动平移。")
        else:
            st.info("请点击上方按钮加载 3D 交互式视图")

