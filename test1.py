# 实验一：八数码问题（A* 启发式搜索算法）
import heapq
import time

def format_state(state):
    """将一维tuple状态格式化为3x3棋盘字符串"""
    s = ""
    for i in range(3):
        for j in range(3):
            v = state[3 * i + j]
            s += "  " if v == 0 else f"{v:2d}"
        s += "\n"
    return s

def inversions(state):
    """计算逆序数，用于判断可解性"""
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def is_solvable(start, goal):
    """
    判断八数码是否可解：
    当且仅当初始状态与目标状态的逆序数奇偶性相同
    """
    return (inversions(start) % 2) == (inversions(goal) % 2)

# ---------------------- 启发函数定义 ----------------------
def misplaced_tiles(state, goal):
    """启发函数1：不在位数 W(n)"""
    return sum(1 for i, v in enumerate(state) if v != 0 and v != goal[i])

def manhattan_distance(state, goal):
    """启发函数2：曼哈顿距离 P(n)"""
    pos = {val: idx for idx, val in enumerate(goal)}
    d = 0
    for idx, v in enumerate(state):
        if v == 0:
            continue
        goal_idx = pos[v]
        x1, y1 = divmod(idx, 3)
        x2, y2 = divmod(goal_idx, 3)
        d += abs(x1 - x2) + abs(y1 - y2)
    return d

# ---------------------- 状态拓展函数 ----------------------
def neighbors_with_moves(state):
    """返回可行的相邻状态及移动方向"""
    zero = state.index(0)
    x, y = divmod(zero, 3)
    moves = []
    for name, (dx, dy) in (("Up", (-1, 0)), ("Down", (1, 0)), ("Left", (0, -1)), ("Right", (0, 1))):
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_idx = nx * 3 + ny
            lst = list(state)
            lst[zero], lst[new_idx] = lst[new_idx], lst[zero]
            moves.append((tuple(lst), name))
    return moves

# ---------------------- A* 启发式搜索算法 ----------------------
def astar(start, goal, heuristic):
    """
    A* 启发式搜索算法实现
    heuristic：可选启发函数，如 misplaced_tiles 或 manhattan_distance
    """
    start_time = time.time()

    if start == goal:
        return {"found": True, "path": [start], "g": 0, "expanded": 0, "generated": 1, "max_frontier": 1, "time": 0.0}

    if not is_solvable(start, goal):
        return {"found": False, "reason": "不可解（初始与目标状态逆序数奇偶性不同）", "time": 0.0}

    # 优先队列（最小堆）
    frontier = []
    counter = 0
    heapq.heappush(frontier, (heuristic(start, goal), counter, start))
    counter += 1

    g_scores = {start: 0}
    parents = {start: None}
    explored = set()
    generated = 1
    expanded = 0
    max_frontier = 1

    while frontier:
        f, _, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        expanded += 1

        if current == goal:
            # 重建路径
            path = []
            s = current
            while s is not None:
                path.append(s)
                s = parents[s]
            path.reverse()
            elapsed = time.time() - start_time
            return {
                "found": True,
                "path": path,
                "g": g_scores[current],
                "expanded": expanded,
                "generated": generated,
                "max_frontier": max_frontier,
                "time": elapsed,
            }

        for nei, mv in neighbors_with_moves(current):
            if nei in explored:
                continue
            tentative_g = g_scores[current] + 1
            if nei not in g_scores or tentative_g < g_scores[nei]:
                g_scores[nei] = tentative_g
                parents[nei] = current
                heapq.heappush(frontier, (tentative_g + heuristic(nei, goal), counter, nei))
                counter += 1
                generated += 1
        max_frontier = max(max_frontier, len(frontier))

    elapsed = time.time() - start_time
    return {"found": False, "reason": "未找到解", "expanded": expanded, "generated": generated, "max_frontier": max_frontier, "time": elapsed}

# ---------------------- 输出结果函数 ----------------------
def print_solution(result):
    """打印搜索结果"""
    if not result["found"]:
        print("未找到解：", result.get("reason", ""))
        return

    print(f"找到解！步数 = {result['g']}")
    print(f"扩展节点数 = {result['expanded']}")
    print(f"生成节点数 = {result['generated']}")
    print(f"最大队列长度 = {result['max_frontier']}")
    print(f"搜索耗时 = {result['time']:.4f} 秒\n")

    print("状态变化路径如下：\n")
    for i, s in enumerate(result["path"]):
        print(f"Step {i}:\n{format_state(s)}")

# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    # 初始状态
    initial = (2, 8, 3,
               1, 6, 4,
               7, 0, 5)
    # 目标状态
    goal = (1, 2, 3,
            8, 0, 4,
            7, 6, 5)

    print("==== 八数码问题启发式搜索实验 ====\n")
    print("初始状态：\n" + format_state(initial))
    print("目标状态：\n" + format_state(goal))
    print(f"逆序数：初始 = {inversions(initial)}, 目标 = {inversions(goal)}")
    print("可解性检测：", "可解" if is_solvable(initial, goal) else "不可解")
    print("----------------------------------------------------")

    print("使用启发函数 W(n) = 不在位数：\n")
    res_w = astar(initial, goal, misplaced_tiles)
    print_solution(res_w)
    print("----------------------------------------------------")

    print("使用启发函数 P(n) = 曼哈顿距离：\n")
    res_p = astar(initial, goal, manhattan_distance)
    print_solution(res_p)

