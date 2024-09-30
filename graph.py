graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 7},
    'C': {'A': 4, 'B': 1, 'D': 3, 'E': 5},
    'D': {'B': 7, 'C': 3, 'E': 2},
    'E': {'C': 5, 'D': 2, 'F': 3},
    'F': {'E': 3}
}

heuristic = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 3,
    'E': 1,
    'F': 0
}
import heapq

def a_star(graph, start, goal, heuristic):
    # Cola de prioridad para la búsqueda
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    # Diccionario para rastrear costos
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    
    # Diccionario para almacenar el camino
    came_from = {}
    
    while open_list:
        current_f_score, current_node = heapq.heappop(open_list)
        
        if current_node == goal:
            # Reconstruir el camino
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            return path[::-1]  # Retornar el camino en el orden correcto
        
        for neighbor, weight in graph[current_node].items():
            tentative_g_score = g_score[current_node] + weight
            
            if tentative_g_score < g_score[neighbor]:
                # Mejor camino encontrado hasta ahora
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic[neighbor]
                heapq.heappush(open_list, (f_score, neighbor))
    
    return None  # No se encontró un camino

# Ejemplo de uso:
start = 'A'
goal = 'F'
path = a_star(graph, start, goal, heuristic)
print("El mejor camino desde", start, "a", goal, "es:", path)
