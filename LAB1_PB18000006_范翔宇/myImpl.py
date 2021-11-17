import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""


def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []


def myBreadthFirstSearch(problem):
    # YOUR CODE HERE
    #跟DFS几乎一样，把堆栈改成队列即可
    visited = {}
    frontier = util.Queue()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))
    #util.raiseNotDefined()
    return []


def myAStarSearch(problem, heuristic):
    # YOUR CODE HERE
    #openlist用优先队列表示 
    openlist = util.PriorityQueue()
    #记录路径
    path = {}
    parent = {}  
    F = {}  
    #初始化 将起点加入open表
    start_state = problem.getStartState()
    parent[start_state] = None
    F[start_state] = heuristic(start_state) + 0 #g(n) = 0
    openlist.update(start_state,F[start_state])
    #开始循环
    while not openlist.isEmpty():
        current_state = openlist.pop()
        G = F[current_state] - heuristic(current_state)
        P = parent[current_state]
        #如果抵达目标状态
        if problem.isGoalState(current_state):
            #记录返回路径
            answer = [current_state]
            while P != None:
                answer.append(P)
                P = path[P]
            return answer[::-1]

        if current_state not in path:
            path[current_state] = P
            for next_state, step_cost in problem.getChildren(current_state):
                F_n = (heuristic(next_state) + step_cost + G)
                if next_state in F and F_n  >= F[next_state] :
                    continue
                else:
                    parent[next_state] = current_state
                    openlist.update(next_state, F_n)
                    F[next_state] = F_n
    if openlist.isEmpty():
        print("error!")
        util.raiseNotDefined()
    return []


"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""

class MyMinimaxAgent():
    
    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if state.isTerminated():
            return None, state.evaluateScore()        

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')
        #personal add here
        # YOUR CODE HERE
        #pacman - MAX
        Flag = state.isMe()
        if Flag == 1:
            depth = depth - 1
            if depth == -1:
                #不能继续递归就返回
                return None, state.evaluateScore()
            for successor in state.getChildren():
                # YOUR CODE HERE
                #调用minmax递归
                tmp, result = self.minimax(successor, depth)
                #pacman对应max
                best_score = max(best_score, result)
                #如果当前result更优
                if best_score == result:
                    best_state = successor
        else:
        #Ghost - MIN
            if depth == -1:
                #同样不能继续递归就返回
                return None, state.evaluateScore()
            for successor in state.getChildren():
                # YOUR CODE HERE
                #调用minmax递归
                tmp, result = self.minimax(successor, depth)
                #ghost对应min
                best_score = min(best_score, result)
                #如果当前result更优
                if best_score == result:
                    best_state = successor

        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state

class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth
    #add here
    #还是仿照minmax样式写比较好，方便递归
    def AlphaBeta(self, state, Alpha, Beta, depth):
        if state.isTerminated():
            return None, state.evaluateScore()

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')


        Flag = state.isMe()
        if Flag:
            #pacman - MAX
            depth = depth - 1
            if depth == -1:
                #不能递归就返回
                return None, state.evaluateScore()
            for successor in state.getChildren():
                # YOUR CODE HERE
                tmp, result = self.AlphaBeta(successor, Alpha, Beta, depth)
                #pacman对应max
                best_score = max(best_score, result)
                #找到对应结果的后继
                if best_score == result:
                    best_state = successor
                #如果大于β就返回结果
                if best_score > Beta:
                    return best_state, best_score
                #否则更新α
                Alpha = max(Alpha, best_score)

        else:
            #ghost - MIN
            if depth == -1:
                #不能递归就返回
                return None, state.evaluateScore()
            for successor in state.getChildren():
                # YOUR CODE HERE
                tmp, result = self.AlphaBeta(successor, Alpha, Beta, depth)
                #ghost对应min
                best_score = min(best_score, result)
                #找到对应结果的后继
                if best_score == result:
                    best_state = successor
                #如果小于α就返回结果
                if best_score < Alpha:
                    return best_state, best_score
                #否则更新β
                Beta = min(Beta, best_score)


        return best_state, best_score

    def getNextState(self, state):
        best_state, best_score = self.AlphaBeta(state, -float('inf'), +float('inf'), self.depth)
        return best_state
        #util.raiseNotDefined()
