# This file is for a global simulation time, which is not used in the current

import threading


class Sim_T(object):
    sim_time = threading.local()

    @staticmethod
    def get_sim_time():
        if not hasattr(Sim_T.sim_time, "current_time"):
            Sim_T.sim_time.current_time = 0
        return Sim_T.sim_time.current_time

    @staticmethod
    def set_sim_time(time):
        Sim_T.sim_time.current_time = time
