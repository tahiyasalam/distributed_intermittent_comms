import numpy as np
from intermittent_comms import Schedule, Robot

if __name__ == "__main__":
    num_teams = 12
    num_robots = 12

    rob_in_teams = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]])
    schedule = Schedule(num_robots, num_teams, rob_in_teams)

    T = schedule.create_teams()

    S = schedule.create_schedule()
    S = np.array([[1, 4, 0, 0], [1, 5, 0, 0], [2, 4, 3, 0], [2, 6, 8, 0], [2, 5, 0, 7], [9, 5, 3, 0], [9, 12, 10, 0], [0, 0, 10, 11], [1, 0, 8, 7], [9, 12, 0, 11], [0, 5, 10, 11], [0, 6, 8,11]])

    sensor_data_period = 20
    eigenvec_data_period = 40

    total_time = 1000

    curr_time = 0

    robots = []
    for r in range(0, num_robots):
        teams = np.where(rob_in_teams[r,:] > 0)[0].astype('int')
        teams = teams + np.ones(np.shape(teams))
        rob = Robot(r + 1, teams, schedule[r])
        robots.append(rob)

    data_val = 1
    while curr_time < total_time:
        # Collect and send sensing data
        for r in range(0, num_robots):
            robots[r].add_new_data([data_val] * sensor_data_period, (curr_time, curr_time + sensor_data_period))
        curr_time += sensor_data_period

        # Use communication protocol here

        # Collect and send eigenvector data
        for r in range(0, num_robots):
            robots[r].add_new_data([data_val * 3] * sensor_data_period, (curr_time, curr_time + eigenvec_data_period))
        curr_time += eigenvec_data_period

