import sys
import time
import argparse
import numpy as np
import torch

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glfw

from model import ActorCritic
from environment import make_env, Obstacle
from util import *

class CrowdSimulator():

    def __init__(self, env, model):

        if not glfw.init():
            print('GLFW initialization failed')
            sys.exit()

        self.env = env
        self.model = model
        
        self.paths = [[] for agent in self.env.agents]

        self.radius = 150
        self.panX = 0
        self.panY = 0
        self.originPanX = 0
        self.originPanY = 0

        self.cp1 = np.zeros(2).astype(np.float32)
        self.cp2 = np.zeros(2).astype(np.float32)

        self.leftMousePressed = False
        self.ctrlPressed = False

        self.panning = False
        self.drawing = False
        
        self.width = 1080
        self.height = 720
        self.window = glfw.create_window(self.width, self.height, 'Crowd Control', None, None)
        if not self.window:
            glfw.terminate()
            sys.exit()

    def initialize(self):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        width, height = glfw.get_window_size(self.window)
        self.reshape(width, height)

    def reshape(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = width / height
        gluPerspective(60, aspect, 1, 1024)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def display(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(-self.panX, self.panY, self.radius,
                    -self.panX, self.panY, 0, 
                    0, 1, 0)
        
        self.paintGround()

        if self.drawing:
            glColor3f(0.1, 0.1, 0.1)
            glRectf(self.cp1[0], self.cp1[1], self.cp2[0], self.cp2[1])

        for obstacle in self.env.obstacles:
            self.paintCircle(obstacle.pos, obstacle.r, [0.2, 0.5, 0.2])
        for i in range(len(self.paths)):
            for p in self.paths[i]:
                self.paintCircle(p, 1, self.env.agents[i].color)
        for agent in self.env.agents:
            self.paintCircle(agent.pos, agent.r, agent.color)
            self.paintCircle(agent.target, 2, [0.0, 0.0, 1.0])
            self.paintArrow(agent.vel, agent.pos, agent.r)

    def paintArrow(self, vel, pos, r):
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_TRIANGLES)
        v1 = normalize(vel)
        glVertex2fv(pos + v1 * r)
        v2 = (rotationMatrix2D(90) @ v1.reshape(2, 1)).T
        glVertex2fv(pos + v2 * r)
        v3 = (rotationMatrix2D(-90) @ v1.reshape(2, 1)).T
        glVertex2fv(pos + v3 * r)
        glEnd()
    
    def paintCircle(self, pos, r, color):
        num = 20
        glColor3fv(color)
        glBegin(GL_TRIANGLE_FAN)

        glVertex2fv(pos)
        for i in range(num + 1):
            x = pos[0] + r * np.cos(i*2*np.pi / num)
            y = pos[1] + r * np.sin(i*2*np.pi / num)
            glVertex2f(x, y)
        glEnd()

        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i in range(num):
            x = pos[0] + r * np.cos(i*2*np.pi / num)
            y = pos[1] + r * np.sin(i*2*np.pi / num)
            glVertex2f(x, y)
        glEnd()
    
    def paintGround(self):
        width = 40
        horizontal = 15
        vertical = 10
        h_size = horizontal * width
        v_size = vertical * width

        glBegin(GL_QUADS)
        for i in range(horizontal):
            for j in range(vertical):
                if (i + j) % 2 == 0:
                    glColor3f(0.5, 0.5, 0.5)
                else:
                    glColor3f(0.25, 0.25, 0.25)
                glVertex2f(i*width-h_size/2, j*width-v_size/2)
                glVertex2f(i*width-h_size/2, (j+1)*width-v_size/2)
                glVertex2f((i+1)*width-(h_size/2), (j+1)*width-v_size/2)
                glVertex2f((i+1)*width-h_size/2, j*width-v_size/2)
        glEnd()

def mouseCallback(window, button, action, mods):
    global sim
    if action == glfw.PRESS:
        if button == glfw.MOUSE_BUTTON_LEFT and sim.ctrlPressed:
            sim.drawing = True
            cursor = glfw.get_cursor_pos(window)
            world = getWorldCoordinate(cursor[0], cursor[1])
            sim.cp1 = np.array([world[0], world[1]])
            sim.cp2 = np.array([world[0], world[1]])
        elif button == glfw.MOUSE_BUTTON_LEFT:
            sim.panning = True
    elif action == glfw.RELEASE:
        if button == glfw.MOUSE_BUTTON_LEFT and sim.drawing:
            sim.drawing = False
            r = min([abs(sim.cp1[0] - sim.cp2[0]), abs(sim.cp1[1] - sim.cp2[1])]) / 2
            if sim.cp1[0] < sim.cp2[0]:
                x = sim.cp1[0] + r
            else:
                x = sim.cp1[0] - r
            if sim.cp1[1] < sim.cp2[1]:
                y = sim.cp1[1] + r
            else:
                y = sim.cp1[1] - r
            sim.env.obstacles.append(Obstacle(x, y, r))
        elif button == glfw.MOUSE_BUTTON_LEFT:
            sim.panning = False

def cursorCallback(window, xpos, ypos):
    global sim
    if sim.panning:
        sim.panX += (xpos - sim.originPanX) / 5
        sim.panY += (ypos - sim.originPanY) / 5
    elif sim.drawing:
        world = getWorldCoordinate(xpos, ypos)
        sim.cp2 = np.array([world[0], world[1]])

    sim.originPanX = xpos
    sim.originPanY = ypos

def scrollCallback(window, xoffset, yoffset):
    global sim
    sim.radius -= yoffset
    sim.radius = clip(sim.radius, 50, 400)

def keyCallback(window, key, scancode, action, mods):
    global sim
    if key == glfw.KEY_LEFT_CONTROL:
        if action == glfw.PRESS:
            sim.ctrlPressed = True
        elif action == glfw.RELEASE:
            sim.ctrlPressed = False

def getWorldCoordinate(x, y):
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    modelView = glGetDoublev(GL_MODELVIEW_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)

    winX = x
    winY = viewport[3] - y
    z = glReadPixels(winX, winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    return gluUnProject(winX, winY, z, modelView, projection, viewport)

def step(env, model, state):
    global avg_step_time, total_steps
    
    state = torch.FloatTensor(state)
    dist, _ = model(state)
    action = dist.sample().numpy()

    start = time.perf_counter()
    next_state, reward, _ = env.step(action)
    elapsed = time.perf_counter() - start
    avg_step_time = avg_step_time + (elapsed - avg_step_time) / total_steps
    print(f'avg step elapsed: {avg_step_time:.5f}', end='\r')
    return next_state, reward

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=".\checkpoints\itr-249,dt-0.25,reward-804.216.dat", help="Model file to load")
    parser.add_argument("--dt", required=True, help="Simulation time step")
    parser.add_argument("-e", "--env", default="basic", help="Environment name to use")
    parser.add_argument("--render", dest="render", action="store_true", help="Render environment")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Do not render environment")
    parser.add_argument("--real", dest="real", action="store_true", help="Render in real time")
    parser.add_argument("--no-real", dest="real", action="store_false", help="Do not render in real time")
    parser.set_defaults(render=True, real=False)
    args = parser.parse_args()
    
    dt = float(args.dt)
    env = make_env(args.env, dt)
    hidden_size = 256

    num_inputs = env.num_observation
    num_outputs = env.num_action
    model = ActorCritic(num_inputs, num_outputs)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    state = env.reset()
    total_reward = 0
    total_steps = 1

    avg_step_time = 0.0

    if args.render:
        sim = CrowdSimulator(env, model)

        glfw.set_cursor_pos_callback(sim.window, cursorCallback)
        glfw.set_key_callback(sim.window, keyCallback)
        glfw.set_mouse_button_callback(sim.window, mouseCallback)
        glfw.set_input_mode(sim.window, glfw.STICKY_MOUSE_BUTTONS, 1)
        glfw.set_scroll_callback(sim.window, scrollCallback)
        glfw.make_context_current(sim.window)
        glfw.swap_interval(0)
        sim.initialize()

        while not glfw.window_should_close(sim.window):
            next_state, reward = step(env, model, state)
            state = next_state
            total_reward += reward
            total_steps += 1

            glfw.poll_events()
            sim.display()
            glfw.swap_buffers(sim.window)

            # record positions for every 5 seconds.
            if (env.frame * env.dt) % 5 == 0:
                for i in range(len(env.agents)):
                    sim.paths[i].append(env.agents[i].pos)
            
            if args.real:
                time.sleep(env.dt)
    else:
        iter = 1000
        for i in range(1000):
            next_state, reward = step(env, model, state)
            state = next_state
            total_reward += reward
            total_steps += 1


    print(f'total steps: {total_steps}, total reward: {total_reward}')
