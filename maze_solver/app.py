
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import networkx as nx
from matplotlib.animation import FuncAnimation
from maze_solver.generator import generate_maze
from algorithms.dfs import dfs_solver
from algorithms.bfs import bfs_solver
from algorithms.astar import astar_solver
import tempfile
import os
import io
import time
from PIL import Image
import matplotlib as mpl

mpl.use('Agg')

st.set_page_config(
    page_title="Maze Solver & Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .stRadio [role=radiogroup] {
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


if "maze" not in st.session_state:
    st.session_state.maze = None
if "solve" not in st.session_state:
    st.session_state.solve = False
if "animation_cache" not in st.session_state:
    st.session_state.animation_cache = {}
if "cached_maze" not in st.session_state:
    st.session_state.cached_maze = None
if "cached_algorithm" not in st.session_state:
    st.session_state.cached_algorithm = None
if "solution_metrics" not in st.session_state:
    st.session_state.solution_metrics = {}
if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = False
if "maze_size" not in st.session_state:
    st.session_state.maze_size = (8, 8)

st.title("🧩 Maze Solver & Visualizer")
st.markdown("""
Explorez comment différents algorithmes de recherche résolvent des labyrinthes.
Cette application visualise DFS, BFS et A* en action avec des animations de l'arbre de recherche.
""")


with st.sidebar:
    st.header("⚙️ Paramètres")
    
    st.subheader("Taille du labyrinthe")
    maze_size = st.slider("Dimensions", min_value=5, max_value=20, value=8, step=1)
    rows, cols = maze_size, maze_size
    
    
    if (rows, cols) != st.session_state.maze_size:
        st.session_state.maze_size = (rows, cols)
        st.session_state.maze = None  
    
    
    st.subheader("Algorithme")
    algorithm = st.radio(
        "Choisir l'algorithme",
        ["DFS", "BFS", "A*", "Comparer tous"],
        index=0,
        help="DFS = profondeur, BFS = largeur, A* = heuristique optimale"
    )
    
    
    st.session_state.comparison_mode = (algorithm == "Comparer tous")
    
    
    st.subheader("Vitesse d'animation")
    animation_speed = st.slider(
        "Choisir la vitesse", 
        min_value=1,
        max_value=10,
        value=3,
        help="1 = lent, 10 = rapide"
    )
    
    
    frame_count = st.slider(
        "Nombre d'images",
        min_value=10,
        max_value=60,
        value=30,
        help="Plus d'images = plus fluide mais plus lent"
    )
    
    
    st.subheader("Actions")
    if st.button("🔄 Générer un nouveau labyrinthe", key="generate_button"):
        with st.spinner("Génération du labyrinthe..."):
            st.session_state.maze = generate_maze(rows, cols)
            st.session_state.solve = False
            st.session_state.animation_cache = {}
            st.session_state.solution_metrics = {}
    
    if st.button("✅ Résoudre le labyrinthe", key="solve_button"):
        st.session_state.solve = True
        
    
    if st.session_state.animation_cache and algorithm in st.session_state.animation_cache:
        st.download_button(
            label="⬇️ Télécharger l'animation GIF",
            data=st.session_state.animation_cache[algorithm],
            file_name=f"maze_solution_{algorithm}.gif",
            mime="image/gif"
        )


if st.session_state.maze is None:
    with st.spinner("Génération du labyrinthe initial..."):
        st.session_state.maze = generate_maze(rows, cols)


cmap = ListedColormap([
    "#FFFFFF",  # 0: white (empty path)
    "#000000",  # 1: black (walls)
    "#FFFACD",  # 2: light yellow (visited cells)
    "#4169E1",  # 3: royal blue (solution path)
    "#32CD32",  # 4: lime green (start)
    "#FF4500",  # 5: orange-red (end)
    "#800080"   # 6: purple (current path)
])

def get_path(parents, start, current):
    """Reconstruct path from parents dictionary"""
    path = []
    while current != start:
        if current not in parents:
            return None
        path.append(current)
        current = parents[current]
    path.append(start)
    path.reverse()
    return path

def solve_maze(algorithm_name, maze):
    """Solve maze with specified algorithm and return results"""
    start_time = time.time()
    
    if algorithm_name == "DFS":
        path, visited, parents = dfs_solver(maze)
    elif algorithm_name == "BFS":
        path, visited, parents = bfs_solver(maze)
    elif algorithm_name == "A*":
        path, visited, parents = astar_solver(maze)
    else:
        return None, None, None, None
    
    end_time = time.time()
    execution_time = round((end_time - start_time) * 1000, 2)  # ms
    
    metrics = {
        "execution_time": execution_time,
        "path_length": len(path),
        "nodes_explored": len(visited),
        "efficiency": round(len(path) / max(1, len(visited)) * 100, 2)
    }
    
    return path, visited, parents, metrics

def create_static_maze_visualization(maze, visited=None, path=None):
    """Create static visualization of maze with optional visited nodes and path"""
    maze_display = np.copy(maze)
    maze_display[maze_display == 1] = 1  # walls
    maze_display[maze_display == 0] = 0  # paths
    
    
    if visited:
        for x, y in visited:
            maze_display[x, y] = 2
    
    
    if path:
        for x, y in path:
            maze_display[x, y] = 3
    
    
    start_node = (1, 1)
    end_node = (maze.shape[0] - 2, maze.shape[1] - 2)
    maze_display[start_node] = 4
    maze_display[end_node] = 5
    
    
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    ax.imshow(maze_display, cmap=cmap)
    
    
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.axis('off')
    
    
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc="#32CD32", label='Départ'),
        plt.Rectangle((0,0),1,1, fc="#FF4500", label='Fin'),
        plt.Rectangle((0,0),1,1, fc="#4169E1", label='Solution'),
        plt.Rectangle((0,0),1,1, fc="#FFFACD", label='Visité'),
        plt.Rectangle((0,0),1,1, fc="#000000", label='Mur')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return fig

def create_animation(maze, algorithm_name, visited, parents, frame_count=30, fps=3):
    """Create animation of search process"""
    try:
        maze = np.copy(maze)
        rows, cols = maze.shape
        start_node = (1, 1)
        end_node = (rows - 2, cols - 2)
        
        
        G = nx.DiGraph()
        G.add_node(start_node, pos=start_node)
        
        for child, parent in parents.items():
            G.add_node(child, pos=child)
            if parent:
                G.add_node(parent, pos=parent)
                G.add_edge(parent, child)
        
       
        pos = nx.get_node_attributes(G, 'pos')
        for node in G.nodes:
            if node not in pos:
                pos[node] = node
        
       
        if len(visited) > frame_count:
            step = len(visited) // frame_count
            sampled_visited = visited[::step][:frame_count]
            if sampled_visited[-1] != visited[-1]:  
                sampled_visited[-1] = visited[-1]
        else:
            sampled_visited = visited
        
        
        fig, (ax_maze, ax_tree) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
        plt.suptitle(f"{algorithm_name} Animation: Arbre de Recherche et Labyrinthe")
        
        nodes = list(G.nodes)
        edges = list(G.edges)
        visited_nodes = [start_node]
        visited_edges = []
        final_path = get_path(parents, start_node, end_node)
        final_path_nodes = set(final_path) if final_path else set()
        
        def update(frame):
            ax_tree.clear()
            ax_maze.clear()

            ax_tree.set_title(f"{algorithm_name} Arbre de Recherche")
            ax_tree.set_axis_off()

            if frame < len(sampled_visited):
                current = sampled_visited[frame]
                visited_nodes.append(current)
                for edge in edges:
                    if edge[1] == current and edge not in visited_edges:
                        visited_edges.append(edge)
            
            
            node_colors = []
            for node in G.nodes:
                if node == start_node:
                    node_colors.append('#32CD32')  # Start - green
                elif node == end_node:
                    node_colors.append('#FF4500')  # End - orange-red
                elif node in visited_nodes:
                    node_colors.append('#FFFACD')  # Visited - light yellow
                else:
                    node_colors.append('#D3D3D3')  # Not visited - light gray
            
            
            edge_colors = []
            for edge in G.edges:
                if edge in visited_edges:
                    edge_colors.append('#000000')  # Visited edge - black
                else:
                    edge_colors.append('#D3D3D3')  # Not visited - light gray
            
            
            nx.draw(
                G, pos, ax=ax_tree, 
                node_color=node_colors, 
                edge_color=edge_colors,
                node_size=50, 
                arrowsize=10,
                width=1.5
            )
            
            
            legend_elements_tree = [
                plt.Rectangle((0,0),1,1, fc="#32CD32", label='Départ'),
                plt.Rectangle((0,0),1,1, fc="#FF4500", label='Fin'),
                plt.Rectangle((0,0),1,1, fc="#FFFACD", label='Visité'),
                plt.Rectangle((0,0),1,1, fc="#D3D3D3", label='Non visité')
            ]
            ax_tree.legend(handles=legend_elements_tree, loc='upper left', bbox_to_anchor=(1.05, 1))
            
            
            ax_maze.set_title("Progression dans le Labyrinthe")
            ax_maze.set_axis_off()

            maze_display_temp = np.copy(maze)
            
            
            for x, y in visited_nodes:
                if (x, y) not in [start_node, end_node]:
                    maze_display_temp[x, y] = 2  # Visited - light yellow

            
            if end_node in visited_nodes:
                solution_path = get_path(parents, start_node, end_node)
                if solution_path:
                    for x, y in solution_path:
                        maze_display_temp[x, y] = 3  # Solution - royal blue
            else:
                
                if frame < len(sampled_visited):
                    current = sampled_visited[frame]
                    current_path = get_path(parents, start_node, current)
                    if current_path:
                        for x, y in current_path:
                            maze_display_temp[x, y] = 6  # Current path - purple

           
            maze_display_temp[1, 1] = 4  # Start - lime green
            maze_display_temp[-2, -2] = 5  # End - orange-red

            
            ax_maze.imshow(maze_display_temp, cmap=cmap)
            ax_maze.set_xticks(np.arange(-.5, cols, 1), minor=True)
            ax_maze.set_yticks(np.arange(-.5, rows, 1), minor=True)
            ax_maze.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
            ax_maze.tick_params(which="minor", size=0)

            
            legend_elements_maze = [
                plt.Rectangle((0,0),1,1, fc="#32CD32", label='Départ'),
                plt.Rectangle((0,0),1,1, fc="#FF4500", label='Fin'),
                plt.Rectangle((0,0),1,1, fc="#4169E1", label='Chemin Solution'),
                plt.Rectangle((0,0),1,1, fc="#800080", label='Chemin Actuel'),
                plt.Rectangle((0,0),1,1, fc="#FFFACD", label='Visité'),
                plt.Rectangle((0,0),1,1, fc="#000000", label='Mur')
            ]
            ax_maze.legend(handles=legend_elements_maze, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        
        anim = FuncAnimation(
            fig, 
            update, 
            frames=len(sampled_visited), 
            interval=1000//fps,  
            repeat=True
        )
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_file:
            anim.save(tmp_file.name, writer='pillow', fps=fps)
            with open(tmp_file.name, 'rb') as f:
                buf = io.BytesIO(f.read())
        os.unlink(tmp_file.name)
        
        plt.close(fig)  
        return buf
        
    except Exception as e:
        st.error(f"Échec de la génération de l'animation : {str(e)}")
        return None

def display_metrics(metrics, algorithm_name):
    """Display metrics for an algorithm solution"""
    if not metrics:
        return
    
    st.subheader(f"📊 Métriques pour {algorithm_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Temps d'exécution", 
            f"{metrics['execution_time']} ms",
            help="Temps de calcul pour trouver la solution"
        )
    
    with col2:
        st.metric(
            "Longueur du chemin", 
            f"{metrics['path_length']} cellules",
            help="Nombre de cellules dans le chemin solution"
        )
    
    with col3:
        st.metric(
            "Cellules explorées", 
            f"{metrics['nodes_explored']}",
            help="Nombre total de cellules visitées pendant la recherche"
        )
    
    with col4:
        st.metric(
            "Efficacité", 
            f"{metrics['efficiency']}%",
            help="Ratio entre longueur du chemin et nombre de cellules explorées"
        )


if st.session_state.comparison_mode:
    
    st.header("🔍 Comparaison des algorithmes")
    
   
    if st.session_state.solve:
        algorithms = ["DFS", "BFS", "A*"]
        
        
        metrics_df = {}
        
        
        for alg in algorithms:
            if alg not in st.session_state.solution_metrics or st.session_state.cached_maze is None or not np.array_equal(st.session_state.cached_maze, st.session_state.maze):
                with st.spinner(f"Résolution avec {alg}..."):
                    path, visited, parents, metrics = solve_maze(alg, st.session_state.maze)
                    st.session_state.solution_metrics[alg] = {
                        "path": path,
                        "visited": visited,
                        "parents": parents,
                        "metrics": metrics
                    }
                    
                    
                    if alg not in st.session_state.animation_cache:
                        with st.spinner(f"Création de l'animation pour {alg}..."):
                            animation = create_animation(
                                st.session_state.maze, 
                                alg, 
                                visited, 
                                parents, 
                                frame_count=frame_count,
                                fps=animation_speed
                            )
                            if animation:
                                st.session_state.animation_cache[alg] = animation
            
           
            if alg in st.session_state.solution_metrics:
                metrics = st.session_state.solution_metrics[alg]["metrics"]
                metrics_df[alg] = metrics
        
        st.session_state.cached_maze = np.copy(st.session_state.maze)
        
        
        st.subheader("📊 Comparaison des performances")
        
       
        metric_cols = st.columns(len(algorithms))
        
        for i, alg in enumerate(algorithms):
            with metric_cols[i]:
                if alg in metrics_df:
                    metrics = metrics_df[alg]
                    st.markdown(f"### {alg}")
                    st.metric("Temps d'exécution", f"{metrics['execution_time']} ms")
                    st.metric("Longueur du chemin", metrics['path_length'])
                    st.metric("Cellules explorées", metrics['nodes_explored'])
                    st.metric("Efficacité", f"{metrics['efficiency']}%")
        
        
        st.subheader("🎬 Animations comparatives")
        
        anim_cols = st.columns(len(algorithms))
        
        for i, alg in enumerate(algorithms):
            with anim_cols[i]:
                st.markdown(f"#### {alg}")
                if alg in st.session_state.animation_cache:
                    st.image(
                        st.session_state.animation_cache[alg],
                        caption=f"Animation {alg}",
                        use_container_width=True
                    )
                else:
                    st.info(f"Animation pour {alg} non disponible")
    else:
        st.info("Cliquez sur 'Résoudre le labyrinthe' pour comparer tous les algorithmes.")
    
    
    with st.expander("Voir le labyrinthe original", expanded=False):
        maze_fig = create_static_maze_visualization(st.session_state.maze)
        st.pyplot(maze_fig, use_container_width=True)

else:
   
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🧩 Labyrinthe")
        
        
        maze_display = np.copy(st.session_state.maze)
        maze_display[maze_display == 1] = 1
        maze_display[maze_display == 0] = 0
        
        path, visited, parents = [], [], {}
        if st.session_state.solve:
            
            if (algorithm not in st.session_state.solution_metrics or 
                st.session_state.cached_maze is None or 
                not np.array_equal(st.session_state.cached_maze, st.session_state.maze)):
                
                with st.spinner(f"Résolution avec {algorithm}..."):
                    path, visited, parents, metrics = solve_maze(algorithm, st.session_state.maze)
                    st.session_state.solution_metrics[algorithm] = {
                        "path": path,
                        "visited": visited,
                        "parents": parents,
                        "metrics": metrics
                    }
                st.session_state.cached_maze = np.copy(st.session_state.maze)
                st.session_state.cached_algorithm = algorithm
            else:
                
                solution_data = st.session_state.solution_metrics[algorithm]
                path = solution_data["path"]
                visited = solution_data["visited"]
                parents = solution_data["parents"]
                metrics = solution_data["metrics"]
            
            
            for x, y in visited:
                maze_display[x, y] = 2
            
           
            for x, y in path:
                maze_display[x, y] = 3
        
        maze_display[1, 1] = 4
        maze_display[-2, -2] = 5
        
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.imshow(maze_display, cmap=cmap)
        
        ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        ax.axis('off')
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc="#32CD32", label='Départ'),
            plt.Rectangle((0,0),1,1, fc="#FF4500", label='Fin'),
            plt.Rectangle((0,0),1,1, fc="#4169E1", label='Solution'),
            plt.Rectangle((0,0),1,1, fc="#FFFACD", label='Visité'),
            plt.Rectangle((0,0),1,1, fc="#000000", label='Mur')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig, use_container_width=True)
        
        if st.session_state.solve and algorithm in st.session_state.solution_metrics:
            metrics = st.session_state.solution_metrics[algorithm]["metrics"]
            display_metrics(metrics, algorithm)
    
    with col2:
        st.header("🎬 Animation")
        if st.session_state.solve and parents:
            if (algorithm not in st.session_state.animation_cache or 
                st.session_state.cached_maze is None or 
                not np.array_equal(st.session_state.cached_maze, st.session_state.maze) or
                st.session_state.cached_algorithm != algorithm):
                
                with st.spinner("Création de l'animation..."):
                    animation = create_animation(
                        st.session_state.maze, 
                        algorithm, 
                        visited, 
                        parents, 
                        frame_count=frame_count,
                        fps=animation_speed
                    )
                    if animation:
                        st.session_state.animation_cache[algorithm] = animation
                        st.session_state.cached_maze = np.copy(st.session_state.maze)
                        st.session_state.cached_algorithm = algorithm
            
            if algorithm in st.session_state.animation_cache:
                st.image(
                    st.session_state.animation_cache[algorithm],
                    caption=f"Animation de {algorithm}: Arbre de Recherche et Labyrinthe",
                    use_container_width=True
                )
            else:
                st.error("Échec de la génération de l'animation.")
        else:
            st.info("Cliquez sur 'Résoudre le labyrinthe' pour voir l'animation de l'arbre de recherche et du labyrinthe.")

st.markdown("---")
st.markdown("""
👨‍💻 **Visualisateur de Labyrinthes** | Explorez les algorithmes de recherche de chemin de manière interactive
""")
