from dataclasses import dataclass, field
import numpy as np
from utils.definitions import ABCAgent, ABCEnvironment
from scipy.spatial import distance


class Environment(ABCEnvironment):

    def __init__(self, map, start, goal):
        self.map = np.array(map)
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.agent_position = np.array(start)
        self.actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]])

    def initial_percepts(self):
        available = []
        for a in self.actions:
            pos = self.start + a
            if (0 <= pos[0] < self.map.shape[0]) and (0 <= pos[1] < self.map.shape[1]) and self.map[pos[0]][
                pos[1]] == 0:
                available.append(pos)

        return {'available_positions': available,
                'position': self.agent_position,
                'goal': self.goal}

    def signal(self, action):

        self.agent_position = action['go_to']

        available = []
        for a in self.actions:
            pos = self.agent_position + a
            if (0 <= pos[0] < self.map.shape[0]) and (0 <= pos[1] < self.map.shape[1]) and self.map[pos[0]][
                pos[1]] == 0:
                available.append(pos)

        return {'available_positions': available,
                'position': self.agent_position,
                'goal': self.goal}


class AgentBFS(ABCAgent):

    def __init__(self, env):

        self.belief_state = env.initial_percepts()
        self.env = env

    def act(self):

        F = [[self.belief_state['position']]]

        while F:
            path = F.pop(0)

            self.belief_state = self.env.signal({'go_to': path[-1]})

            if (path[-1] == self.belief_state['goal']).all():
                return path
            else:
                for p in self.belief_state['available_positions']:
                    F.append(path + [p])

        return []


class AgentDFS(ABCAgent):

    def __init__(self, env):

        self.belief_state = env.initial_percepts()
        self.env = env

    def act(self):

        F = [[self.belief_state['position']]]

        while F:
            path = F.pop(0)

            self.belief_state = self.env.signal({'go_to': path[-1]})

            if (path[-1] == self.belief_state['goal']).all():
                return path
            else:
                for p in self.belief_state['available_positions']:
                    # Checks whether a cycle will be made
                    makes_cycle = False
                    for pos in path:
                        if (pos == p).all():
                            makes_cycle = True
                            break

                    if not makes_cycle:
                        F = [path + [p]] + F

        return []


class GreedyAgent(ABCAgent):
    def __init__(self, env):
        self.belief_state = env.initial_percepts()
        self.frontier = [[self.belief_state['position']]]
        self.visited = []

    def act(self):
        """Implements the agent action
        """

        # Select a path from the frontier
        cost = 999999
        j = 0
        k = 0

        for i in self.frontier:
            if distance.euclidean(i[-1], self.belief_state['goal']) < cost:
                j = k
                cost = distance.euclidean(i[-1], self.belief_state['goal'])
            k = k + 1

        # Path with the lowest value of heuristic function of frontier
        path = self.frontier.pop(j)

        # Visit the last node in the path
        action = {'go_to': path[-1]}
        # The agente sends a position and the full path to the environment
        self.belief_state = self.env.signal(action)

        # Add visited node
        self.visited.append(path[-1])

        # From the list of viable available_positions given by the environment
        # Select a random neighbor that has not been visited yet

        viable_neighbors = self.belief_state['available_positions']

        # If the agent is not stuck
        if viable_neighbors:
            for neighbor in viable_neighbors:

                # Multiple-Path Pruning
                exist = False

                for node in self.visited:
                    if (node == neighbor).all():
                        exist = True
                        break

                # If node is not in self.visited
                if exist == False:
                    self.frontier = [path + [neighbor]] + self.frontier

    def run(self):
        """Keeps the agent acting until it finds the goal
        """

        # Run agent
        while (self.belief_state['position'] != self.belief_state['goal']).any() and self.frontier:
            self.act()

        print(self.belief_state['position'])


class AStarAgent(ABCAgent):
    def __init__(self, env):
        self.env = env
        self.belief_state = env.initial_percepts()
        self.frontier = [[self.belief_state['position']]]
        self.cost = [0]
        # Initializes list of visited nodes for multiple path prunning
        self.visited = []

    def act(self):
        """Implements the agent action
        """

        # Select a path from the frontier
        cost = 999999
        j = 0
        k = 0

        for i in self.frontier:
            if self.cost[k] + distance.euclidean(i[-1], self.belief_state['goal']) < cost:
                j = k
                cost = self.cost[k] + distance.euclidean(i[-1], self.belief_state['goal'])
            k = k + 1

        # Path with the lowest value of cost + heuristic function of frontier
        path = self.frontier.pop(j)
        cost = self.cost.pop(j)

        # Visit the last node in the path
        action = {'go_to': path[-1]}
        # Send position
        self.belief_state = self.env.signal(action)

        # Add visited node
        self.visited.append(path[-1])

        # From the list of viable available_positions given by the environment
        # Select a random neighbor that has not been visited yet

        viable_neighbors = self.belief_state['available_positions']

        # If the agent is not stuck
        if viable_neighbors:
            for neighbor in viable_neighbors:

                # Multiple-Path Pruning
                exist = False

                for node in self.visited:
                    if (node == neighbor).all():
                        exist = True
                        break

                # If node is not in self.visited
                if exist == False:
                    self.frontier = [path + [neighbor]] + self.frontier
                    self.cost = [cost + distance.euclidean(path[-1], neighbor)] + self.cost

    def run(self):
        """Keeps the agent acting until it finds the goal
        """

        # Run agent
        while (self.belief_state['position'] != self.belief_state['goal']).any() and self.frontier:
            self.act()
        print(self.belief_state['position'])


if __name__ == "__main__":
    map = [[0, 0, 1],
           [1, 0, 1],
           [1, 0, 0]]

    env = Environment(map, [0, 0], [2, 2])

    ag = AStarAgent(env)

    ag.run()

    GreedyAgent(env).run()
