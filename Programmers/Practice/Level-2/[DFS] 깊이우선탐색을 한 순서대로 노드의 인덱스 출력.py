# DFS: 노드의 탐색 순서 == 스택에 들어간 순서 == 방문처리한 순서

# E - D-F
#   |-A-C
#   |-|-B

tree = {
        'A': ['B', 'C', 'E'],
        'B': ['A'],
        'C': ['A'],
        'D': ['E', 'F'],
        'E': ['A', 'D'],
        'F': ['D']
}
visited = []
def dfs(graph, start):
    # visited = []
    visited.append(start)
    for i in reversed(graph[start]):
        if i not in visited:
            dfs(graph, i)
    return visited
    
print(dfs(tree, 'E'))

visited2 = []
def dfs(graph, start):
    # visited = []
    visited2.append(start)
    for i in graph[start]:
        if i not in visited2:
            dfs(graph, i)
    return visited2
    
print(dfs(tree, 'E'))

graph = {
        'A': set(['B', 'C', 'E']),
        'B': set(['A']),
        'C': set(['A']),
        'D': set(['E', 'F']),
        'E': set(['A', 'D']),
        'F': set(['D'])
}

def dfs(graph, start):
    visited = []
    stack = [start]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            stack += graph[n] - set(visited)
    return visited

print(dfs(graph, 'E'))