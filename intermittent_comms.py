import numpy as np


class Robot:
    def __init__(self, ID, teams, schedule):
        self.ID = ID
        self.teams = teams
        self.schedule = schedule
        self.sensor_data = {}
        self.eigen_data = {}

    def add_new_data(self, new_data, time, data_type):
        t_start, t_end = time
        if data_type == 'sensor':
            self.sensor_data[(t_start, t_end, self.ID)] = new_data
        else:
            self.eigen_data[(t_start, t_end, self.ID)] = new_data

    def get_data_from_robots(self, new_data, data_type):
        if data_type == 'sensor':
            self.sensor_data = {**self.sensor_data, **new_data}
        else:
            self.eigen_data = {**self.eigen_data, **new_data}

    def construct_data_matrix(self):
        return None



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
            T[i] += np.ones(np.shape(T[i]))
        return T

    def create_schedule(self):
        """Create schedule based on team compositions
        return team adjacency matrix """
        T = self.create_teams()
        schedule = np.zeros((self.num_robots, self.num_teams))
        teams = np.where(self.rob_in_teams[0,:] > 0)[0].astype('int')
        teams = teams + np.ones(np.shape(teams))
        teams = teams.astype('int')

        schedule[0, 0:np.shape(teams)[0]] = teams

        for j in range(1, self.num_robots):
            teams = np.where(self.rob_in_teams[j,:] > 0)[0].astype('int')
            teams = teams + np.ones(np.shape(teams))
            teams = teams.astype('int')

            for t in range(0, np.shape(teams)[0]):
                rule12 = False
                rule3 = False
                team = teams[t]

                for col in range(0, self.num_teams):
                    comp = (team == schedule[:, col])
                    # print(col, team)
                    # print(schedule[:, col])
                    # print(comp)
                    if team in schedule[:, col]:
                        # print(team)
                        schedule[j, col] = team
                        rule12 = True
                        break
            if not rule12:
                col = 0
                while col <= self.num_teams and not rule3:
                    placed_teams = np.unique(schedule[np.where(schedule[:, col] > 0), col]).astype('int')
                    sum_t = 0
                    # print(placed_teams)
                    for pt in range(0, np.shape(placed_teams)[0]):
                        pteam = placed_teams[pt].astype('int')
                        if np.intersect1d(T[team - 1], T[pteam - 1]).size == 0:
                            sum_t += 1
                    if sum_t == np.shape(placed_teams)[0]:
                        schedule[j, col] = team
                        # print(team)
                        rule3 = True
                    col += 1
                # print(j, col)
            #
            # print(schedule, rule12, rule3)
        # schedule = schedule[~np.all(schedule == 0, axis=0)]
        return schedule





