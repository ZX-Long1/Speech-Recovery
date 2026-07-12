# ============================================================
# 实验二：迷宫问题 A*算法求解最短路径
# ============================================================
import heapq
import math

# -----------------------------
# 一、迷宫图定义
# -----------------------------
graph = {
    (1,1): [(1,2)],
    (1,2): [(1,1), (1,3)],
    (1,3): [(2,3)],
    (1,4): [(2,4)],
    (2,1): [(3,1), (2,2)],
    (2,2): [(2,1), (3,2), (2,3)],
    (2,3): [(2,2), (1,3), (2,4)],
    (2,4): [(2,3), (1,4), (3,4)],
    (3,1): [(2,1), (4,1), (3,2)],
    (3,2): [(3,1), (2,2), (3,3)],
    (3,3): [(3,2), (4,3), (3,4)],
    (3,4): [(3,3), (2,4)],
    (4,1): [(3,1), (4,2)],
    (4,2): [(4,1), (4,3)],
    (4,3): [(4,2), (3,3), (4,4)],
    (4,4): [(4,3)]
}

start = (1,1)
goal = (4,4)

# -----------------------------
# 二、启发函数定义
# -----------------------------
def h_manhattan(node, goal=(4,4)):
    """h(n) = (X - x) + (Y - y)"""
    x, y = node
    X, Y = goal
    return (X - x) + (Y - y)

def h_euclidean(node, goal=(4,4)):
    """欧几里得距离"""
    x, y = node
    X, Y = goal
    return math.sqrt((X - x)**2 + (Y - y)**2)

def h_zero(node, goal=(4,4)):
    """零启发"""
    return 0

# -----------------------------
# 三、A*算法实现
# -----------------------------
def astar(graph, start, goal, hfunc):
    open_heap = []
    heapq.heappush(open_heap, (hfunc(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded_order = []

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded_order.append(current)

        if current == goal:
            # 回溯路径
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, expanded_order

        for neighbor in graph.get(current, []):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + hfunc(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return None, expanded_order

# -----------------------------
# 四、打印迷宫网格
# -----------------------------
def print_grid(path):
    path_set = set(path) if path else set()
    for i in range(1,5):
        for j in range(1,5):
            pos = (i,j)
            if pos == start:
                print("S", end=" ")
            elif pos == goal:
                print("G", end=" ")
            elif pos in path_set:
                print("*", end=" ")
            elif pos in graph or any(pos in v for v in graph.values()):
                print(".", end=" ")
            else:
                print("#", end=" ")
        print()
    print()

# -----------------------------
# 五、主程序执行部分
# -----------------------------
if __name__ == "__main__":
    print("======== A*算法求解迷宫最短路径 ========")

    # 使用曼哈顿距离启发函数
    path, expanded = astar(graph, start, goal, h_manhattan)

    print("\n节点展开顺序：")
    print(" -> ".join(str(n) for n in expanded))
    print("\n最短路径：")
    print(" -> ".join(str(p) for p in path))
    print("\n迷宫路径图：")
    print_grid(path)

    # 启发函数比较实验
    print("======== 启发函数比较 ========")
    for name, func in [("曼哈顿距离", h_manhattan),
                       ("欧几里得距离", h_euclidean),
                       ("零启发(迪杰斯特拉)", h_zero)]:
        p, exp = astar(graph, start, goal, func)
        print(f"{name}: 展开节点数 = {len(exp)}, 路径长度 = {len(p)}")
