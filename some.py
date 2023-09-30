import heapq

def dijkstra(graph, start, end):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # This condition might improve efficiency by avoiding the unnecessary processing of longer paths
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex  # Store the path
                heapq.heappush(priority_queue, (distance, neighbor))

    # Reconstruct the path from end to start by following the predecessors
    path = []
    while end:
        path.append(end)
        end = predecessors[end]
    path.reverse()

    return distances, path

graph = {
    'Entrance': {'Room1': 5, 'Room2': 10},
    'Room1': {'Entrance': 5, 'Room2': 3, 'Room3': 7},
    'Room2': {'Entrance': 10, 'Room1': 3, 'Room4': 2},
    'Room3': {'Room1': 7},
    'Room4': {'Room2': 2}
}

distances, path = dijkstra(graph, 'Entrance', 'Room3')
print("Shortest Distance:", distances['Room3'])
print("Optimal Path:", path)
