import numpy as np


def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()


# TODO: double-check to make sure this works
def find_nearest_below(my_array, target):
    diff = my_array - target
    mask = np.ma.greater_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmax()


class Robot:
    objs = []  # Registrar
    discretization = (600, 600)

    def __init__(self, ID, teams, schedule):
        self.ID = ID
        self.teams = teams
        self.schedule = schedule
        self.active_locations = {} # Store active location as indexed by (start_time, end_time): locations
        self.sensor_data = {} # Store data as (start_time, end_time): data
        self.eigen_data = {}
        Robot.objs.append(self)

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

    @classmethod
    def construct_data_matrix(cls):
        from operator import itemgetter
        max_t = 0
        # Find end time of data matrix
        for obj in cls.objs:
            keys = obj.sensor_data.keys()
            max_from_keys = max(keys, key=itemgetter(2))[0]  # Returns largest end time
            if max_t < max_from_keys:
                max_t = max_from_keys

        data_matrix = np.zeros((Robot.discretization[0], Robot.discretization[1], max_t))

        for obj in cls.objs:  # Fill in data matrix
            active_location_times = obj.active_locations.keys()
            active_location_start = active_location_times[0]
            active_location_end = active_location_times[1]

            # Match sensor data start and end time to active locations
            for key, data in obj.sensor_data:
                data_start_t, data_end_t = key[0], key[1]

                data_matrix[obj.active_locations[0], obj.active_locations[1], data_start_t:data_end_t] = data

        return data_matrix


class Schedule:
    def __init__(self, num_robots, num_teams, rob_in_teams):
        self.num_robots = num_robots
        self.num_teams = num_teams
        self.rob_in_teams = rob_in_teams

    def create_teams(self):
        """Create random teams based on number of robots and number of teams"""
        T = [[] for x in range(self.num_teams)]

        for i in range(0, self.num_teams):
            T[i] = np.where(self.rob_in_teams[:, i] > 0)
            T[i] += np.ones(np.shape(T[i]))
        return T

    def create_schedule(self):
        """Create schedule based on team compositions"""
        T = self.create_teams()
        schedule = np.zeros((self.num_robots, self.num_teams))
        teams = np.where(self.rob_in_teams[0,:] > 0)[0].astype('int')
        teams = teams + np.ones(np.shape(teams))
        teams = teams.astype('int')

        schedule[0, 0:np.shape(teams)[0]] = teams

        for j in range(1, self.num_robots):
            teams = np.where(self.rob_in_teams[j, :] > 0)[0].astype('int')
            teams = teams + np.ones(np.shape(teams))
            teams = teams.astype('int')

            for t in range(0, np.shape(teams)[0]):
                rule12 = False
                rule3 = False
                team = teams[t]

                for col in range(0, self.num_teams):
                    if team in schedule[:, col]:
                        schedule[j, col] = team
                        rule12 = True
                        break
                if not rule12:
                    col = 0
                    while col <= self.num_teams and not rule3:
                        placed_teams = np.unique(schedule[np.where(schedule[:, col] > 0), col]).astype('int')
                        sum_t = 0
                        for pt in range(0, np.shape(placed_teams)[0]):
                            pteam = placed_teams[pt].astype('int')
                            if np.intersect1d(T[team - 1], T[pteam - 1]).size == 0:
                                sum_t += 1
                        if sum_t == np.shape(placed_teams)[0]:
                            schedule[j, col] = team
                            rule3 = True
                        col += 1

        schedule = schedule[:, ~np.all(schedule == 0, axis=0)]  # Remove columns full of zeros
        return schedule
