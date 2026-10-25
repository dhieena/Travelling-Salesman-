import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="TSP Solver", layout="wide")

# --- Default Data ---
DEFAULT_CITIES = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", "Alexandria", "Berlin", "Tokyo", "Rio", "Budapest"]
DEFAULT_X = [0, 3, 6, 9, 15, 10, 16, 5, 8, 1.5]
DEFAULT_Y = [1, 2, 1, 4.5, -1, 2.5, 11, 4, 9, 12]

CITY_ICONS = {
    "Gliwice": "♕", "Cairo": "♖", "Rome": "♗", "Krakow": "♘", "Paris": "♙",
    "Alexandria": "♔", "Berlin": "♚", "Tokyo": "♛", "Rio": "♜", "Budapest": "♝"
}

# --- Helper Functions ---
def plot_city_network(city_coords):
    cities = list(city_coords.keys())
    colors = sns.color_palette("pastel", len(cities))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(False)

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i % len(colors)]
        icon = CITY_ICONS.get(city, "📍")
        
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=30, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=10, ha='center', va='bottom', xytext=(0, -25), textcoords='offset points')

        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.15)

    return fig

def dist_two_cities(c1_coords, c2_coords):
    return np.sqrt(np.sum((np.array(c1_coords) - np.array(c2_coords)) ** 2))

def total_dist_individual(individual, city_coords):
    total_dist = 0
    n = len(individual)
    for i in range(n):
        c1, c2 = individual[i], individual[(i + 1) % n]
        total_dist += dist_two_cities(city_coords[c1], city_coords[c2])
    return total_dist

def run_ga(cities_names, city_coords, n_population=250, n_generations=20, crossover_per=0.8, mutation_per=0.2):
    population = []
    for _ in range(n_population):
        ind = list(cities_names).copy()
        random.shuffle(ind)
        population.append(ind)

    for _ in range(n_generations):
        distances = np.array([total_dist_individual(ind, city_coords) for ind in population])
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        fitness = max_dist - distances + 1e-6
        fitness_probs = fitness / np.sum(fitness)

        num_parents = int(crossover_per * n_population)
        if num_parents % 2 != 0:
            num_parents -= 1
            
        cumsum = np.cumsum(fitness_probs)
        parents = []
        for _ in range(num_parents):
            r = random.random()
            idx = np.searchsorted(cumsum, r)
            parents.append(population[min(idx, len(population) - 1)])

        offspring_list = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i+1]
            cut = random.randint(1, max(1, len(p1) - 2))
            
            o1 = p1[:cut] + [c for c in p2 if c not in p1[:cut]]
            o2 = p2[:cut] + [c for c in p1 if c not in p2[:cut]]

            if random.random() < mutation_per and len(o1) > 1:
                idx1, idx2 = random.sample(range(len(o1)), 2)
                o1[idx1], o1[idx2] = o1[idx2], o1[idx1]

            if random.random() < mutation_per and len(o2) > 1:
                idx1, idx2 = random.sample(range(len(o2)), 2)
                o2[idx1], o2[idx2] = o2[idx2], o2[idx1]

            offspring_list.extend([o1, o2])

        combined = parents + offspring_list
        comb_distances = np.array([total_dist_individual(ind, city_coords) for ind in combined])
        sorted_indices = np.argsort(comb_distances)
        population = [combined[i] for i in sorted_indices[:n_population]]

    best_route = population[0]
    best_dist = total_dist_individual(best_route, city_coords)
    return best_route, best_dist

# ==========================================
# SINGLE PAGE INTERFACE
# ==========================================
st.title("Traveling Salesman Problem (TSP) Solver")

# 1. Show Initial Full Network Diagram
st.subheader("Initial City Network")
default_coords = dict(zip(DEFAULT_CITIES, zip(DEFAULT_X, DEFAULT_Y)))
fig_initial = plot_city_network(default_coords)
st.pyplot(fig_initial)

st.divider()

# 2. City Coordinate Input Form
st.subheader("Customize City Coordinates")
with st.form("city_input_form"):
    updated_city_coords = {}
    
    for i in range(10):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            city_name = st.text_input(
                label=f"City {i+1}", 
                value=DEFAULT_CITIES[i] if i < len(DEFAULT_CITIES) else f"City {i+1}",
                key=f"city_name_{i}"
            )
        with col2:
            x_val = st.number_input(
                label=f"x-coordinate ({city_name})", 
                value=float(DEFAULT_X[i]) if i < len(DEFAULT_X) else 0.0,
                step=0.5,
                key=f"x_coord_{i}"
            )
        with col3:
            y_val = st.number_input(
                label=f"y-coordinate ({city_name})", 
                value=float(DEFAULT_Y[i]) if i < len(DEFAULT_Y) else 0.0,
                step=0.5,
                key=f"y_coord_{i}"
            )

        if city_name.strip():
            updated_city_coords[city_name.strip()] = (x_val, y_val)

    submitted = st.form_submit_button("Submit")

# 3. Calculate and Render Best Route on Submit
if submitted:
    st.divider()
    st.subheader("Optimized TSP Route Results")
    
    cities_list = list(updated_city_coords.keys())
    
    if len(cities_list) >= 2:
        best_route, min_dist = run_ga(cities_list, updated_city_coords)
        
        x_shortest = [updated_city_coords[c][0] for c in best_route] + [updated_city_coords[best_route[0]][0]]
        y_shortest = [updated_city_coords[c][1] for c in best_route] + [updated_city_coords[best_route[0]][1]]

        fig_route, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5, markersize=8)

        for idx, txt in enumerate(best_route):
            ax.annotate(f"{idx+1}- {txt}", (x_shortest[idx], y_shortest[idx]), fontsize=12)

        plt.title("TSP Best Route using Genetic Algorithm", fontsize=16)
        plt.suptitle(f"Total Distance: {round(min_dist, 3)}", fontsize=12)
        st.pyplot(fig_route)
    else:
        st.warning("Please specify at least 2 cities to compute a route.")
