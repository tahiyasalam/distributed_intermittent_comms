import numpy as np


class Robot:
    def __init__(self, ID, teams, schedule):
        self.ID = ID
        self.teams = teams
        self.schedule = schedule
        self.data = {}

    def add_new_data(self, new_data, time):
        t_start, t_end = time
        self.data[(t_start, t_end, self.ID)] = new_data

    def get_data_from_robots(self, new_data):
        self.data = {**self.data, **new_data}


class Schedule:
    def __init__(self, num_robots, num_teams, rob_in_teams):
        self.num_robots = num_robots
        self.num_teams = num_teams
        self.rob_in_teams = rob_in_teams

    def create_teams(self):
        """Create random teams based on number of robots and number of teams"""
        T = [[] for x in range(self.num_teams)]

        for i in range(0, self.num_teams):
            T[i] = np.where(self.rob_in_teams[:,i] > 0)
        return T

    def create_schedule(self):
        """Create schedule based on team compositions
        return team adjacency matrix """
        T = self.create_teams()
        schedule = np.zeros((self.num_robots, self.num_teams))
        teams = np.where(self.rob_in_teams[0,:] > 0)[0]
        schedule[0, 0:np.shape(teams)[0]] = teams + 1

        for j in range(1, self.num_robots):
            teams = np.where(self.rob_in_teams[j,:] > 0)[0]

            for t in range(0, np.shape(teams)[0]):
                rule12 = False
                rule3 = False
                team = teams[t]
                for col in range(0, self.num_teams):
                    comp = (team == schedule[:, col])
                    print(col)
                    print(schedule[:, col])
                    if np.any(comp):
                        # print(team)
                        schedule[j, col] = team + 1
                        rule12 = True
                        break
            if not rule12:
                col = 0
                while col <= self.num_teams and not rule3:
                    placed_teams = np.unique(schedule[np.where(schedule[:, col] > 0), col]).astype('int')
                    sum_t = 0
                    # print(placed_teams)
                    for pt in range(0, np.shape(placed_teams)[0]):
                        pteam = placed_teams[pt]
                        print(pteam)
                        if np.intersect1d(T[team - 1], T[pteam - 1]).any():
                            sum_t += 1
                    if sum_t == np.shape(placed_teams)[0]:
                        schedule[j, col] = team + 1
                        print(team)
                        rule3 = True
                    col += 1
        # schedule = schedule[~np.all(schedule == 0, axis=0)]
        return schedule





