import time
import numpy as np
import sys
import math
from itertools import combinations
from xml.etree.ElementTree import parse
from scipy.spatial.distance import pdist, cdist

# import ext_state

env_list = ['basic', 'circle1', 'circle2', 'crossing1', 'crossing2', 'obstacles']

def make_env(name, dt):
    env_fname = './envs/' + name + '.xml'
    if name in env_list:
        return Environment(name, env_fname, dt)
    return None

def make_env_pool(dt):
    env_pool = []
    for env in env_list:
        if env is not 'basic':
            env_pool.append(make_env(env, dt))
    return env_pool

class Environment():

    def __init__(self, name, env_fname, dt):
        self.tree = parse(env_fname)
        self.name = name

        self.eps = 2 # goal distance between agent and target
        self.dt = dt # timestep

        self.v_min = 0.2 # desired minimum velocity
        self.v_max = 1.2 # desired maximum velocity
        self.t_max = 30 # maximum sight of agent

        self.w_min = 0.0 # desired minimum angular delta
        self.w_max = np.pi / 4 # desired maximum angular delta
        
        self.w1 = 4.0 # distance reward weight
        self.w2 = 3.0 # collision reward weight
        self.w3 = 2.0 # velocity flood reward weight
        self.w4 = 1.0 # theta flood reward weight

        self.n_ray = 20 # number of rays
        self.d_past = 3 # past depth map size
        self.d_future = 3 # future depth map size
        self.d_total = self.d_past + self.d_future + 1 # total depth map size

        self.min_collision_reward = -10

        self.avg_times = np.zeros(10, dtype=np.float64)

        # self.reset()
        self.num_observation = 3 + self.n_ray * (self.d_total + 2) # number of states
        self.num_action = 2 # number of actions

    def reset(self):
        root = self.tree.getroot()

        self.agents = []
        for agent in root.findall('agent'):
            x = float(agent.findtext('x'))
            y = float(agent.findtext('y'))
            x1 = float(agent.findtext('x1'))
            y1 = float(agent.findtext('y1'))
            r = float(agent.findtext('radius'))
            color_r = float(agent.findtext('color_R')) / 255
            color_g = float(agent.findtext('color_G')) / 255
            color_b = float(agent.findtext('color_B')) / 255
            color = [color_r, color_g, color_b]
            self.agents.append(Agent(x, y, x1, y1, r, color))

        self.obstacles = []
        for obstacle in root.findall('obstacle'):
            x = float(obstacle.findtext('x'))
            y = float(obstacle.findtext('y'))
            r = float(obstacle.findtext('radius'))
            self.obstacles.append(Obstacle(x, y, r))
        
        self.n_agent = len(self.agents)
        # for vectorization
        self.p_t = np.array([agent.pos for agent in self.agents]) # positions
        self.v_t = np.array([agent.vel for agent in self.agents]) # velocities
        self.w_t = np.array([agent.theta for agent in self.agents]) # angles
        self.o_t = np.array([agent.ori for agent in self.agents]) # orientations
        self.targets = np.array([agent.target for agent in self.agents])
        self.obs_pos = np.array([obstacle.pos for obstacle in self.obstacles])
        self.dones = np.array([False for agent in self.agents])
        self.force = np.zeros((len(self.agents), 2, 1), dtype=np.float64)
        self.p_t1 = self.v_t1 = self.w_t1 = self.o_t1 = None

        self.frame = 1
        self.n_agent = len(self.agents)

        # use current state as memory
        ext_state = self.externalStates(self.p_t, self.v_t, self.w_t, self.o_t)
        depth_maps = []
        for i in range(len(self.agents)):
            depth_maps.append([ext_state[i]] * self.d_total)
        self.depth_maps = np.array(depth_maps)

        return self.computeStates()

    # action : [len(agents), 2] force
    def step(self, action):
        # 1. apply forces
        # start = time.perf_counter()
        self.force = action.reshape(len(self.agents), 2, 1)
        self.p_t1, self.v_t1, self.w_t1, self.o_t1 = self.nextStates(self.p_t, self.v_t, self.w_t, self.o_t, self.force)
        self.updateStates()
        # elapsed = time.perf_counter() - start
        # self.avg_times[0] += (elapsed - self.avg_times[0]) / self.frame

        # 2. compute rewards
        # start = time.perf_counter()
        rewards = self.computeRewards()
        # elapsed = time.perf_counter() - start
        # self.avg_times[1] += (elapsed - self.avg_times[1]) / self.frame

        self.p_t, self.v_t, self.w_t, self.o_t = self.p_t1, self.v_t1, self.w_t1, self.o_t1

        # 3. compute states
        # start = time.perf_counter()
        states = self.computeStates()
        # elapsed = time.perf_counter() - start
        # self.avg_times[2] += (elapsed - self.avg_times[2]) / self.frame

        dist = np.linalg.norm(self.targets - self.p_t1, axis=1)
        self.dones = np.logical_or(self.dones, dist < self.eps)
        
        self.frame += 1
        
        # print('update:', self.avg_times[0])
        # print('compute reward:', self.avg_times[1])
        # print('compute state:', self.avg_times[2])
        # print('internal state:', self.avg_times[4])
        # print('external state:', self.avg_times[5])

        return states, rewards, self.dones
    
    def nextStates(self, pos, vel, theta, ori, force):
        # update positions, velocities
        new_pos = pos + self.dt * vel
        new_vel = vel + self.dt * (ori @ force).reshape(self.n_agent, 2)

        # update orientations
        def rotationMatrix(c, s):
            return np.array([[c, -s], [s, c]])

        new_theta = np.arctan2(new_vel[:, 1], new_vel[:, 0])
        cos = np.cos(new_theta)
        sin = np.sin(new_theta)

        new_ori = np.array([rotationMatrix(cos[i], sin[i]) for i in range(self.n_agent)])

        return new_pos, new_vel, new_theta, new_ori

    def updateStates(self):
        for i in range(len(self.agents)):
            self.agents[i].pos = self.p_t1[i]
            self.agents[i].vel = self.v_t1[i]
            self.agents[i].theta = self.w_t1[i]
            self.agents[i].ori = self.o_t1[i]

    def computeRewards(self):

        distance_reward = self.distanceRewards()
        collision_reward, has_collision = self.collisionRewards()
        velocity_reward = self.velocityRewards()
        orientation_reward= self.orientationRewards()

        distance_reward *= np.logical_not(has_collision)
        
        # done_reward = np.full(n, -1, dtype=np.float64)
        # done_reward[self.dones] = 0

        if self.frame % 500 == 0:
            print('frame:', self.frame)
            print('distance_reward:', distance_reward)
            print('collision_reward:', collision_reward)
            print('velocity_reward:', velocity_reward)
            print('orientation reward:', orientation_reward)
            # print('done reward:', done_reward)
            print()
        
        return self.w1 * distance_reward + self.w2 * collision_reward + self.w3 * velocity_reward + self.w4 * orientation_reward

    def distanceRewards(self):
        # 1. distance reward(continuous)
        dist1 = np.linalg.norm(self.targets - self.p_t, axis=1)
        dist2 = np.linalg.norm(self.targets - self.p_t1, axis=1)
        distance_reward = dist1 - dist2

        # discrete distance reward
        # distance_reward = np.zeros(len(self.agents), dtype=np.float64)
        # distance_reward[dist < self.eps] = 1
        return distance_reward
    
    def collisionRewards(self):
        collision_reward = np.zeros(self.n_agent, dtype=np.float64)
        has_collision = np.zeros(self.n_agent)
        objs = np.concatenate((self.agents, self.obstacles))
        if len(self.obstacles) > 0:
            obj_pos = np.concatenate((self.p_t, self.obs_pos), axis=0)
        else:
            obj_pos = self.p_t
        for i in range(self.n_agent):
            for j in range(len(obj_pos)):
                if i == j:
                    continue
                d = np.linalg.norm(self.p_t[i] - obj_pos[j])
                if d - (self.agents[i].r + objs[j].r) < 0:
                    collision_reward[i] += self.min_collision_reward
                    has_collision[i] = True
                    break
        
        return collision_reward, has_collision
    
    def velocityRewards(self):
        velocity_reward = np.zeros(self.n_agent, dtype=np.float64)
        # velocity = np.linalg.norm(new_vel, axis=1)
        for i in range(self.n_agent):
            v = np.sqrt(np.dot(self.agents[i].vel, self.agents[i].vel))
            velocity_reward[i] = -self.flood(v, self.v_min, self.v_max)

        return velocity_reward

    def orientationRewards(self):
        orientation_reward = np.zeros(self.n_agent, dtype=np.float64)
        ori_diff = np.abs(self.w_t1 - self.w_t)
        for i in range(self.n_agent):
            orientation_reward[i] = -self.flood(ori_diff[i], self.w_min, self.w_max)

        return orientation_reward

    def computeStates(self):
        # start = time.perf_counter()
        int_state = self.internalStates()
        # elapsed = time.perf_counter() - start
        # self.avg_times[4] += (elapsed - self.avg_times[4]) / self.frame

        # start = time.perf_counter()
        ext_state = self.externalStates(self.p_t, self.v_t, self.w_t, self.o_t)
        v_x_maps, v_y_maps = self.velocityMaps(self.p_t, self.v_t, self.w_t, self.o_t)
        # elapsed = time.perf_counter() - start
        # self.avg_times[5] += (elapsed - self.avg_times[5]) / self.frame

        # expected external states
        pos, vel, theta, ori = self.p_t, self.v_t, self.w_t, self.o_t
        for i in reversed(range(0, self.d_future)):
            new_pos, new_vel, new_theta, new_ori = self.nextStates(pos, vel, theta, ori, self.force)
            new_ext_state = self.externalStates(new_pos, new_vel, new_theta, new_ori)
            self.depth_maps[:,i,:] = new_ext_state
            pos, vel, theta, ori = new_pos, new_vel, new_theta, new_ori

        for i in reversed(range(self.d_future, self.d_total)):
            self.depth_maps[:,i,:] = self.depth_maps[:,i-1,:]
        self.depth_maps[:,self.d_future - 1,:] = ext_state
        
        return np.concatenate((int_state, self.depth_maps.reshape(-1, self.n_ray * self.d_total), v_x_maps, v_y_maps), axis=1)

    # states : list of [pos, |vel|] -> [len(agents), 3]
    def internalStates(self):
        n = len(self.agents)
        m_inv = np.linalg.inv(self.o_t)
        pos_diff = (self.targets - self.p_t).reshape(n, 2, 1)
        relative_pos = (m_inv @ pos_diff).reshape(n, 2)
        v = np.linalg.norm(self.v_t, axis=1).reshape(n, 1)
        state_int = np.concatenate((relative_pos, v), axis=1)
        return state_int

    def externalStates(self, pos, vel, theta, ori):
        dw = np.array([(i*np.pi) / (self.n_ray-1) - np.pi/2 for i in range(self.n_ray)])
        thetas = np.array([w + dw for w in theta])
        d = np.empty((self.n_agent, self.n_ray, 2))
        d[:,:,0] = np.cos(thetas)
        d[:,:,1] = np.sin(thetas)

        objs = np.concatenate((self.agents, self.obstacles))
        if len(self.obstacles) > 0:
            obj_pos = np.concatenate((pos, self.obs_pos), axis=0)
        else:
            obj_pos = self.p_t
        p = np.array([[obj - agent for obj in obj_pos] for agent in pos])

        state_ext=[]
        for i in range(self.n_agent):
            depth_map = []
            for j in range(self.n_ray):
                dij = d[i][j]
                t_min = self.t_max
                for k in range(len(obj_pos)):
                    if i == k:
                        continue
                    pik = p[i][k]
                    tm = np.dot(pik, dij)
                    lm_2 = np.dot(pik, pik) - tm ** 2
                    dt = objs[k].r ** 2 - lm_2
                    if dt > 0:
                        dt = np.sqrt(dt)
                        t0 = tm - dt
                        t1 = tm + dt
                        if t0 > 0:
                            t_min = min(t_min, t0)
                        elif t1 > 0:
                            t_min = min(t_min, t1)
                depth_map.append(t_min)
            state_ext.append(depth_map)
        return np.array(state_ext)
    
    def velocityMaps(self, pos, vel, theta, ori):
        dw = np.array([(i*np.pi) / (self.n_ray-1) - np.pi/2 for i in range(self.n_ray)])
        thetas = np.array([w + dw for w in theta])
        d = np.empty((self.n_agent, self.n_ray, 2))
        d[:,:,0] = np.cos(thetas)
        d[:,:,1] = np.sin(thetas)

        objs = np.concatenate((self.agents, self.obstacles))
        if len(self.obstacles) > 0:
            obj_pos = np.concatenate((pos, self.obs_pos), axis=0)
        else:
            obj_pos = self.p_t
        p = np.array([[obj - agent for obj in obj_pos] for agent in pos])

        v_x_maps = []
        v_y_maps = []
        for i in range(self.n_agent):
            v_x_map = []
            v_y_map = []
            for j in range(self.n_ray):
                dij = d[i][j]
                t_min = self.t_max
                idx_min = -1    
                for k in range(len(obj_pos)):
                    if i == k:
                        continue
                    pik = p[i][k]
                    tm = np.dot(pik, dij)
                    lm_2 = np.dot(pik, pik) - tm ** 2
                    dt = objs[k].r ** 2 - lm_2
                    if dt > 0:
                        dt = np.sqrt(dt)
                        t0 = tm - dt
                        t1 = tm + dt
                        if t0 > 0 and t0 < t_min:
                            t_min = t0
                            idx_min = k
                        elif t1 > 0 and t1 < t_min:
                            t_min = t1
                            idx_min = k
                if idx_min == -1 : # not found
                    v_x_map.append(0.)
                    v_y_map.append(0.)
                else:
                    ori_inv = np.linalg.inv(ori[i])
                    v1 = vel[i]
                    if idx_min < self.n_agent:
                        v2 = vel[idx_min]
                    else:
                        v2 = np.array([0, 0], dtype=np.float64)
                    vel_diff = (v2 - v1).reshape(2, 1)
                    relative_vel = (ori_inv @ vel_diff).reshape(2)
                    v_x_map.append(relative_vel[0])
                    v_y_map.append(relative_vel[1])
            v_x_maps.append(v_x_map)
            v_y_maps.append(v_y_map)
        return np.array(v_x_maps), np.array(v_y_maps)

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def flood(self, value, v_min, v_max):
        return abs(np.min([value - v_min, 0])) + abs(max([value - v_max, 0]))

class Agent():

    def __init__(self, x, y, x1, y1, r, color):
        
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.target = np.array([x1, y1], dtype=np.float64)
        self.r = r
        self.color = color

        direction = self.target - self.pos
        self.theta = math.atan2(direction[1], direction[0])
        cos = math.cos(self.theta)
        sin = math.sin(self.theta)
        self.ori = np.array([[cos, -sin], [sin, cos]])

class Obstacle():

    def __init__(self, x, y, r):
        self.pos = np.array([x, y], dtype=np.float64)
        self.r = r

if __name__ == "__main__":
    env = make_env('basic')
    state = env.reset()
    state_int = env.internalStates()
    state_ext = env.externalStates()
    state = np.concatenate((state_int, state_ext), axis=1)
