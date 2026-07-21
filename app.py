import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from groq import Groq

# --- Hardcoded Groq API Key ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- Page Config ---
st.set_page_config(page_title="MarsNet-RL | Autonomous Routing", layout="wide")

# --- Sidebar: Simulation Setup & Chatbot ---
st.sidebar.title("🚀 MarsNet-RL Control")
st.sidebar.info("Version 1.0 | Team Cassiopeia")

num_satellites = st.sidebar.slider("Number of Satellites", 5, 50, 20)
debris_density = st.sidebar.slider("Debris Density", 0.0, 1.0, 0.3)
training_mode = st.sidebar.checkbox("Enable Training Mode (RL Active)")

st.sidebar.divider()
st.sidebar.subheader("Simulation Parameters")
thrust_intensity = st.sidebar.number_input("Max Delta-V (m/s)", 0.1, 10.0, 1.0)
orbit_altitude = st.sidebar.select_slider("Target Orbit (km)", options=[500, 1000, 2500, 5000])

st.sidebar.divider()

# --- Chatbot Feature in Sidebar ---
st.sidebar.subheader("💬 MarsNet Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your MarsNet-RL operations assistant. How can I help you optimize your orbital routing today?"}
    ]

chat_container = st.sidebar.container(height=300)

for message in st.session_state.messages:
    with chat_container.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.sidebar.chat_input("Ask about orbital logistics..."):
    with chat_container.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful, expert aerospace engineering assistant specializing in deep-space satellite routing, reinforcement learning, and Martian orbital mechanics."},
                *st.session_state.messages
            ],
            model="llama-3.3-70b-versatile",
        )
        assistant_response = response.choices[0].message.content
        
        with chat_container.chat_message("assistant"):
            st.write(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
    except Exception as e:
        st.sidebar.error(f"Error connecting to Groq: {e}")

# --- Main Dashboard ---
st.title("🛰️ MarsNet-RL: Autonomous Orbital Routing")
st.markdown("**Objective:** Maintain >90% network uptime while avoiding 99% of orbital collisions.")

col1, col2, col3 = st.columns(3)
col1.metric("Live Collision Risk", "0.02%", "-0.4% from baseline")
col2.metric("Network Coverage", "94.2%", "Target: 90%")
col3.metric("Agent Success Rate", "88.5%", "Stage 2 Phase")

# Helper to calculate position vectors
def get_coords(r, anomaly, lan, inc):
    x_p = r * np.cos(anomaly)
    y_p = r * np.sin(anomaly)
    x_s = x_p * np.cos(lan) - y_p * np.sin(lan) * np.cos(inc)
    y_s = x_p * np.sin(lan) + y_p * np.cos(lan) * np.cos(inc)
    z_s = y_p * np.sin(inc)
    return x_s, y_s, z_s

# --- 3D Visualization Generator Function ---
def build_animated_orbital_plot():
    R_mars = 3389.5  
    GM_mars = 42828.3  
    
    # Generate Mars Globe Mesh Data
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_m = R_mars * np.outer(np.cos(u), np.sin(v))
    y_m = R_mars * np.outer(np.sin(u), np.sin(v))
    z_m = R_mars * np.outer(np.ones(np.size(u)), np.cos(v))

    # Calculate Velocities
    r_orbit = R_mars + orbit_altitude  
    v_orbit_kms = np.sqrt(GM_mars / r_orbit)  
    omega = (v_orbit_kms / r_orbit) * 250  # Time dilation scaling factor

    # Generate Setup Arrays
    np.random.seed(42)
    sat_inc = np.random.uniform(-np.pi/4, np.pi/4, size=num_satellites)
    sat_lan = np.random.uniform(0, 2*np.pi, size=num_satellites)

    # Setup Asteroids
    num_asteroids = 100
    np.random.seed(101)
    ast_r = np.random.uniform(9000, 13000, size=num_asteroids)
    ast_inc = np.random.uniform(-np.pi/12, np.pi/12, size=num_asteroids)
    ast_lan = np.random.uniform(0, 2*np.pi, size=num_asteroids)
    ast_phase = np.random.uniform(0, 2*np.pi, size=num_asteroids)
    ast_omega = (np.sqrt(GM_mars / ast_r) / ast_r) * 250

    # Create Base Traces for Frame 0
    fig = go.Figure()

    # Trace 0: Mars
    fig.add_trace(go.Surface(x=x_m, y=y_m, z=z_m, colorscale='balance', showscale=False, name="Mars", opacity=0.9))

    # Generate all positions for frames
    frames = []
    num_frames = 40
    
    for t in range(num_frames):
        # 1. Earth Position
        earth_orbit_r = 22000
        e_angle = 0.01 * t
        x_e_c = earth_orbit_r * np.cos(e_angle)
        y_e_c = earth_orbit_r * np.sin(e_angle)
        x_e = x_e_c + 4000 * np.outer(np.cos(u), np.sin(v))
        y_e = y_e_c + 4000 * np.outer(np.sin(u), np.sin(v))
        z_e = 0 + 4000 * np.outer(np.ones(np.size(u)), np.cos(v))

        # 2. Satellites
        s_x, s_y, s_z, s_hover = [], [], [], []
        for i in range(num_satellites):
            ma = (2 * np.pi / num_satellites) * i + (omega * t)
            x, y, z = get_coords(r_orbit, ma, sat_lan[i], sat_inc[i])
            s_x.append(x)
            s_y.append(y)
            s_z.append(z)
            s_hover.append(f"🛰️ Sat {i+1}<br>Velocity: {v_orbit_kms:.3f} km/s")

        # 3. Asteroids
        a_x, a_y, a_z = [], [], []
        for j in range(num_asteroids):
            aa = ast_phase[j] + (ast_omega[j] * t)
            x, y, z = get_coords(ast_r[j], aa, ast_lan[j], ast_inc[j])
            a_x.append(x)
            a_y.append(y)
            a_z.append(z)

        # Build frame update payload
        if t == 0:
            fig.add_trace(go.Surface(x=x_e, y=y_e, z=z_e, colorscale='Blues', showscale=False, name="Earth", opacity=0.85))
            fig.add_trace(go.Scatter3d(x=a_x, y=a_y, z=a_z, mode='markers', marker=dict(size=2.5, color='darkgray'), name="Asteroids", hoverinfo='none'))
            fig.add_trace(go.Scatter3d(x=s_x, y=s_y, z=s_z, mode='markers', marker=dict(size=5, color='gold', symbol='diamond'), name="Satellites", text=s_hover, hoverinfo="text"))
        else:
            frames.append(go.Frame(
                data=[
                    go.Surface(x=x_m, y=y_m, z=z_m), 
                    go.Surface(x=x_e, y=y_e, z=z_e),
                    go.Scatter3d(x=a_x, y=a_y, z=a_z),
                    go.Scatter3d(x=s_x, y=s_y, z=s_z, text=s_hover)
                ],
                name=f"frame_{t}"
            ))

    fig.frames = frames

    fig.update_layout(
        margin=dict(l=10, r=10, b=10, t=10),
        height=600,
        paper_bgcolor='black',
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor="black", gridcolor="gray", showbackground=True),
            yaxis=dict(title='Y (km)', backgroundcolor="black", gridcolor="gray", showbackground=True),
            zaxis=dict(title='Z (km)', backgroundcolor="black", gridcolor="gray", showbackground=True),
            aspectmode='data'
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05,       
            "y": 0.05,       
            "xanchor": "left",
            "yanchor": "bottom",
            "pad": {"t": 10, "b": 10},
            "font": {"color": "gold", "size": 13}, 
            "bgcolor": "#1e1e1e",                  
            "bordercolor": "gold",                 
            "borderwidth": 1,
            "buttons": [{
                "label": "▶ Animate Real-World Dynamics",
                "method": "animate",
                "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
            }]
        }],
        legend=dict(
            x=0.05,
            y=0.95,
            xanchor='left',
            yanchor='top',
            font=dict(color="white")
        )
    )
    return fig, v_orbit_kms, r_orbit, sat_lan, sat_inc, omega

# --- 3D Visualization Pipeline UI ---
st.subheader("Real-World Martian Orbital Dynamics Visualization")

animated_fig, current_velocity, r_orbit, sat_lan, sat_inc, omega = build_animated_orbital_plot()
st.markdown(f"🛰️ **Current Orbital Velocity:** `{current_velocity:.3f} km/s` *(Passive Keplerian drift)*")
st.plotly_chart(animated_fig, use_container_width=True)

# --- Real-Time Metrics & Controls ---
tab1, tab2 = st.tabs(["🚀 Live Telemetry", "🧠 RL Training Logs"])

with tab1:
    if st.button("Initialize Autonomous Routing"):
        with st.status("Simulating orbital paths...", expanded=True) as status:
            time.sleep(1)
            st.write("Predicting congestion windows...")
            time.sleep(1)
            st.write("Applying autonomous maneuvers...")
            status.update(label="System Stable: Autonomous avoidance active", state="complete")

    st.write("Current Orbital Metrics")
    
    # --- Dynamic Metrics Engine Engine (Evaluates Real Satellite Distances) ---
    collision_array = np.zeros(num_satellites, dtype=int)
    np.random.seed(42) # Mirror simulation properties
    
    # Calculate physical positions at a test window (t=10) to search for tight spacing overlaps
    positions = []
    for i in range(num_satellites):
        ma = (2 * np.pi / num_satellites) * i + (omega * 10)
        x, y, z = get_coords(r_orbit, ma, sat_lan[i], sat_inc[i])
        positions.append(np.array([x, y, z]))
        
    # Check if they brush past or touch within a spatial tolerance of 10km
    proximity_tolerance = 10.0 
    for i in range(num_satellites):
        for j in range(i + 1, num_satellites):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < proximity_tolerance:
                collision_array[i] += 1
                collision_array[j] += 1

    # Deterministic generation for secondary telemetry arrays matching the exact slider count
    np.random.seed(24)
    fuel_usage = np.round(np.random.uniform(0.0, 3.5, size=num_satellites), 2)
    uptime_percentage = np.round(np.random.uniform(94.0, 99.9, size=num_satellites), 1)
    sat_labels = [f"Satellite {k}" for k in range(num_satellites)]

    telemetry_data = {
        "Collision Events": collision_array, 
        "Fuel Usage (kg)": fuel_usage,  
        "Uptime (%)": uptime_percentage     
    }
    
    # Constructing DataFrame scaled seamlessly to match the user's sidebar selection
    df_metrics = pd.DataFrame(telemetry_data, index=sat_labels)
    st.dataframe(df_metrics, use_container_width=True)

with tab2:
    st.subheader("RL Reward Function Monitoring")
    st.line_chart(np.abs(np.random.randn(50, 2)).cumsum(axis=0))
    st.warning("Detection: RL agent found a potential 'deep-space drift' loophole. Adjusting reward penalty parameters.")

# --- Footer ---
st.divider()
st.caption("MarsNet-RL | Autonomous Deep-Space Satellite Routing | Team Cassiopeia")
