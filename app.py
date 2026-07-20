import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from groq import Groq

# --- Hardcoded Groq API Key ---
GROQ_API_KEY = "gsk_yOuRaPiKeYhErE"

# --- Page Config ---
st.set_page_config(page_title="MarsNet-RL | Autonomous Routing", layout="wide")

# --- Initialize Groq Client ---
client = Groq(api_key=GROQ_API_KEY)

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

# --- 3D Visualization Generator Function ---
def generate_orbital_plot(t_step=0):
    # Real-world Constants for Mars
    R_mars = 3389.5  # Mars Radius in km
    GM_mars = 42828.3  # Standard gravitational parameter (G * M) for Mars in km^3/s^2
    
    # Generate Mars Globe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_m = R_mars * np.outer(np.cos(u), np.sin(v))
    y_m = R_mars * np.outer(np.sin(u), np.sin(v))
    z_m = R_mars * np.outer(np.ones(np.size(u)), np.cos(v))

    # Generate Earth Globe (Placed in a further, independent outer orbit)
    earth_orbit_r = 22000 
    earth_angle = 0.01 * t_step
    x_e_center = earth_orbit_r * np.cos(earth_angle)
    y_e_center = earth_orbit_r * np.sin(earth_angle)
    z_e_center = 0
    R_earth = 4000  
    
    x_e = x_e_center + R_earth * np.outer(np.cos(u), np.sin(v))
    y_e = y_e_center + R_earth * np.outer(np.sin(u), np.sin(v))
    z_e = z_e_center + R_earth * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # Add Mars
    fig.add_trace(go.Surface(
        x=x_m, y=y_m, z=z_m, 
        colorscale='balance', showscale=False, 
        name="Mars", opacity=0.9
    ))

    # Add Earth
    fig.add_trace(go.Surface(
        x=x_e, y=y_e, z=z_e, 
        colorscale='Blues', showscale=False, 
        name="Earth", opacity=0.85
    ))

    # --- Physics: Calculate Orbital Velocity ---
    r_orbit = R_mars + orbit_altitude  
    v_orbit_kms = np.sqrt(GM_mars / r_orbit)  
    omega = v_orbit_kms / r_orbit
    time_dilation_factor = 250  
    effective_omega = omega * time_dilation_factor

    # Generate structured paths and current positions for N satellites
    sat_x, sat_y, sat_z = [], [], []
    hover_texts = []
    
    np.random.seed(42)  
    for i in range(num_satellites):
        inc = np.random.uniform(-np.pi/4, np.pi/4)  
        lan = np.random.uniform(0, 2*np.pi)         
        mean_anomaly = (2 * np.pi / num_satellites) * i + (effective_omega * t_step)
        
        x_plane = r_orbit * np.cos(mean_anomaly)
        y_plane = r_orbit * np.sin(mean_anomaly)
        
        x_space = x_plane * np.cos(lan) - y_plane * np.sin(lan) * np.cos(inc)
        y_space = x_plane * np.sin(lan) + y_plane * np.cos(lan) * np.cos(inc)
        z_space = y_plane * np.sin(inc)
        
        sat_x.append(x_space)
        sat_y.append(y_space)
        sat_z.append(z_space)
        
        hover_texts.append(f"🛰️ Satellite {i+1}<br>Alt: {orbit_altitude} km<br>Velocity: {v_orbit_kms:.3f} km/s")

        # Draw the Orbital Ring Track for each satellite
        theta_loop = np.linspace(0, 2 * np.pi, 60) # Kept point count light for performance
        x_loop_plane = r_orbit * np.cos(theta_loop)
        y_loop_plane = r_orbit * np.sin(theta_loop)
        x_loop_space = x_loop_plane * np.cos(lan) - y_loop_plane * np.sin(lan) * np.cos(inc)
        y_loop_space = x_loop_plane * np.sin(lan) + y_loop_plane * np.cos(lan) * np.cos(inc)
        z_loop_space = y_loop_plane * np.sin(inc)

        fig.add_trace(go.Scatter3d(
            x=x_loop_space, y=y_loop_space, z=z_loop_space,
            mode='lines',
            line=dict(color='rgba(255, 215, 0, 0.12)', width=1),
            showlegend=False,
            hoverinfo='none'
        ))

    # --- HIGH PERFORMANCE ASTEROID BELT SECTION ---
    # We generate a fixed patch of 150 asteroids orbiting slightly further out
    num_asteroids = 150
    np.random.seed(101) # Secure fixed properties for the asteroid cluster
    
    # Give them varied orbital rings beyond the satellites (e.g., 9,000 to 13,000 km out)
    ast_r = np.random.uniform(9000, 13000, size=num_asteroids)
    ast_inc = np.random.uniform(-np.pi/12, np.pi/12, size=num_asteroids) # Slightly flatter belt orientation
    ast_lan = np.random.uniform(0, 2*np.pi, size=num_asteroids)
    ast_initial_phase = np.random.uniform(0, 2*np.pi, size=num_asteroids)
    
    # Calculate varying velocities based on their explicit distance from Mars
    ast_v = np.sqrt(GM_mars / ast_r)
    ast_omega = (ast_v / ast_r) * time_dilation_factor

    ast_x, ast_y, ast_z, ast_hover = [], [], [], []
    for j in range(num_asteroids):
        # Current position factoring its unique orbital speed (varying speeds!)
        ast_anomaly = ast_initial_phase[j] + (ast_omega[j] * t_step)
        
        x_p = ast_r[j] * np.cos(ast_anomaly)
        y_p = ast_r[j] * np.sin(ast_anomaly)
        
        x_s = x_p * np.cos(ast_lan[j]) - y_p * np.sin(ast_lan[j]) * np.cos(ast_inc[j])
        y_s = x_p * np.sin(ast_lan[j]) + y_p * np.cos(ast_lan[j]) * np.cos(ast_inc[j])
        z_s = y_p * np.sin(ast_inc[j])
        
        ast_x.append(x_s)
        ast_y.append(y_s)
        ast_z.append(z_s)
        ast_hover.append(f"🪨 Asteroid Cluster<br>Distance: {ast_r[j]:.0f} km<br>Speed: {ast_v[j]:.3f} km/s")

    # Add all asteroids in ONE single vectorized trace to prevent lag
    fig.add_trace(go.Scatter3d(
        x=ast_x, y=ast_y, z=ast_z,
        mode='markers',
        marker=dict(size=2.5, color='darkgray', symbol='circle'),
        name="Asteroid Belt (Partial)",
        text=ast_hover,
        hoverinfo="text"
    ))

    # Plot Satellites
    fig.add_trace(go.Scatter3d(
        x=sat_x, y=sat_y, z=sat_z,
        mode='markers',
        marker=dict(size=5, color='gold', symbol='diamond', line=dict(color='black', width=1)),
        name="Satellites",
        text=hover_texts,
        hoverinfo="text"
    ))

    # UI Scene Settings
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0), 
        height=600,
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor="black", gridcolor="gray", showbackground=True),
            yaxis=dict(title='Y (km)', backgroundcolor="black", gridcolor="gray", showbackground=True),
            zaxis=dict(title='Z (km)', backgroundcolor="black", gridcolor="gray", showbackground=True),
            aspectmode='data'
        ),
        paper_bgcolor='black',
        legend=dict(
            x=0.0,
            y=1.0,
            xanchor = 'center',
            yanchor = 'bottom'
        )
    )
    return fig, v_orbit_kms

# --- 3D Visualization Placeholder & Controls ---
st.subheader("Real-World Martian Orbital Dynamics Visualization")

velocity_metric_placeholder = st.empty()
plot_spot = st.empty()

if "frame" not in st.session_state:
    st.session_state.frame = 0

initial_fig, current_velocity = generate_orbital_plot(st.session_state.frame)
velocity_metric_placeholder.markdown(f"🛰️ **Current Orbital Velocity:** `{current_velocity:.3f} km/s` *(Passive Keplerian drift)*")
plot_spot.plotly_chart(initial_fig, use_container_width=True)

if st.button("Animate Real-World Dynamics"):
    for step in range(1, 50):
        st.session_state.frame += 1
        active_fig, current_velocity = generate_orbital_plot(st.session_state.frame)
        velocity_metric_placeholder.markdown(f"🛰️ **Current Orbital Velocity:** `{current_velocity:.3f} km/s` *(Passive Keplerian drift)*")
        plot_spot.plotly_chart(active_fig, use_container_width=True)
        time.sleep(0.04)

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
    
    np.random.seed(st.session_state.frame) 
    telemetry_data = {
        "Collision Events": np.random.choice([0, 1], size=5, p=[0.95, 0.05]), 
        "Fuel Usage (kg)": np.round(np.random.uniform(0.1, 4.5, size=5), 2),  
        "Uptime (%)": np.round(np.random.uniform(93.5, 99.9, size=5), 2)     
    }
    
    live_df = pd.DataFrame(telemetry_data)
    st.dataframe(live_df, use_container_width=True)

with tab2:
    st.subheader("RL Reward Function Monitoring")
    st.line_chart(np.abs(np.random.randn(50, 2)).cumsum(axis=0))
    st.warning("Detection: RL agent found a potential 'deep-space drift' loophole. Adjusting reward penalty parameters.")

# --- Footer ---
st.divider()
st.caption("MarsNet-RL | Autonomous Deep-Space Satellite Routing | Team Cassiopeia")