# BFS: 노드의 탐색 순서 == 큐에 들어간 순서 == 방문처리한 순서

# E - D-F
#   |-A-C
#   |-|-B

graph = {'A': set(['B','C','E']),\
        'B':set(['A']),\
        'C':set(['A']),\
        'D':set(['E','F']),\
        'E':set(['A','D']),\
        'F':set(['D'])}

# 리스트로 하면?
graph2 = {'A': ['B','C','E'],\
        'B':['A'],\
        'C':['A'],\
        'D':['E','F'],\
        'E':['A','D'],\
        'F':['D']}

from collections import deque

def bfs(graph, start):
    # 현재 노드(시작start 노드)
    q = deque([start])
    # 방문 처리용 리스트
    visited = []
    # 큐에 아무 것도 없을 때까지:
    while q:
        # 큐의 가장 앞 노드를 꺼낸다. 
        front_node = q.popleft()
        # print(front_node, end=' ')
        
        # 해당 노드front_node 와 연결된 노드들을 순회하면서
        # 처음 방문한 노드면 (if not visited)
        if front_node not in visited:
            # (1) 방문 처리 
            # (2) 큐에 넣음
            visited.append(front_node)
            q += graph[front_node] - set(visited)
    return visited

def bfs2(graph, start):
    q = deque([start])
    visited = []
    
    while q:
        front_node = q.popleft()
        if front_node not in visited:
            visited.append(front_node)
            q += set(graph2[front_node]) - set(visited)
    return visited

print(bfs(graph, 'E'))
print(bfs2(graph2,'E'))
