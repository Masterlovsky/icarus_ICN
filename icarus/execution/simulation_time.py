# This file is for a global simulation time, which is not used in the current

class Sim_T(object):
    sim_time = 0

    @staticmethod
    def get_sim_time():
        return Sim_T.sim_time

    @staticmethod
    def set_sim_time(time):
        Sim_T.sim_time = time
