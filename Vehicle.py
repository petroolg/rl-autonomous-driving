import numpy as np

from RigidBody import RigidBody, rot


class Wheel:

    def __init__(self, pos, rad):
        self.attach_point = pos
        self.m_fwd_axis = []
        self.m_side_axis = []
        self.m_wheel_torque = 0.0
        self.m_wheel_speed = 0.0
        self.m_wheel_inertia = rad*rad
        self.m_wheel_radius = rad
        self.set_steering_angle(0)

    def set_steering_angle(self, angle):
        #fwd and side vector
        vec = np.array([[[0.0], [1.0]],[[-1.0],[0.0]]])
        vec = rot(angle).dot(vec)

        self.m_fwd_axis = vec[0]
        self.m_side_axis = vec[1]

    def add_transm_torque(self, new_value):
        self.m_wheel_torque += new_value

    def calculate_force(self, relative_ground_speed, timestep):
        patch_speed = -self.m_fwd_axis*self.m_wheel_speed*self.m_wheel_radius
        vel_diff = relative_ground_speed + patch_speed

        side_vel = vel_diff.T.dot(self.m_side_axis) * self.m_side_axis
        forw_mag = vel_diff.T.dot(self.m_fwd_axis)
        fwd_vel = forw_mag * self.m_fwd_axis

        #friction forces
        resp_force = -side_vel * 2.0
        resp_force -= fwd_vel

        self.m_wheel_torque += forw_mag * self.m_wheel_radius
        self.m_wheel_speed += timestep * self.m_wheel_torque / self.m_wheel_inertia
        self.m_wheel_torque = 0

        return resp_force


class Vehicle(RigidBody):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wheels = [] #type: list[Wheel]
        width = 10.0
        length = 14.0
        self.wheels.append(Wheel(np.array([[width], [length]]), 0.5))
        self.wheels.append(Wheel(np.array([[-width], [length]]), 0.5))
        self.wheels.append(Wheel(np.array([[width], [-length]]), 0.5))
        self.wheels.append(Wheel(np.array([[-width], [-length]]), 0.5))
        self.last_err = 0
        self.y_vel_ref = kwargs.get('y_vel_ref', 0)
        self.line_ref = kwargs.get('line_ref', 1)

    def set_steering(self, steering):
        steeringLock = 0.5

        #apply steering angle to front wheels
        self.wheels[2].set_steering_angle(-steering * steeringLock)
        self.wheels[3].set_steering_angle(-steering * steeringLock)

    @property
    def line(self):
        if self.x < -20:
            return -1
        elif self.x > 20:
            return 1
        return 0

    def set_throttle(self, throttle, all_wheels=False):
        torque = 10.0
        if all_wheels:
            self.wheels[0].add_transm_torque(throttle * torque)
            self.wheels[1].add_transm_torque(throttle * torque)

        self.wheels[2].add_transm_torque(throttle * torque)
        self.wheels[3].add_transm_torque(throttle * torque)

    def set_brakes(self, brakes):
        brake_torque = 4
        for w in self.wheels:
            w.add_transm_torque(-w.m_wheel_speed * brake_torque * brakes)

    def update(self, timestep):

        for i, w in enumerate(self.wheels):
            world_wheel_offset = super().rel_to_world(w.attach_point)
            world_ground_vel = super().point_vel(world_wheel_offset)
            rel_ground_speed = super().world_to_rel(world_ground_vel)
            rel_resp_force = w.calculate_force(rel_ground_speed, timestep)
            world_resp_force = super().rel_to_world(rel_resp_force)
            # print('Force:', i, world_resp_force, 'offset:', world_wheel_offset)
            super().add_force(world_resp_force, world_wheel_offset)

        super().update(timestep)
        # pygame.time.wait(1)

    def keep_line(self, targ_line=0, targ_vel=0):
        targ_vel = self.y_vel_ref if targ_vel == 0 else targ_vel
        targ_line = self.line_ref if targ_line == 0 else targ_line
        throttle, steering, brakes = 0, 0, 0
        ref = 30 if targ_line == 1 else -30
        err = ref - self.m_pos[0][0]
        derr = err - self.last_err
        steering = err * 0.03 + derr*0.9
        # print(err)
        err_vel = (abs(np.linalg.norm(self.m_vel)) - abs(targ_vel))[0]

        steering = np.sign(steering) if abs(steering) > 1 else steering
        if err_vel > 0:
            brakes = err_vel * 0.1
        else:
            throttle = -err_vel * 3

        throttle = 20 if abs(throttle) > 20 else throttle
        brakes = 1 if abs(brakes) > 1 else brakes

        self.last_err = err
        return throttle, steering, brakes
