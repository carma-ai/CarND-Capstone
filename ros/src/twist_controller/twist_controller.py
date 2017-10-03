from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
STEER_WEIGHT = 2.0


class Controller(object):
    def __init__(self, throttle_params, brake_params, wheel_base,
                 steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        # Initialize all the PIDs
        self.throttle_pid = PID(
            throttle_params['kp'],
            throttle_params['ki'],
            throttle_params['kd'],
            throttle_params['min'],
            throttle_params['max'])
        self.brake_pid = PID(
            brake_params['kp'],
            brake_params['ki'],
            brake_params['kd'],
            brake_params['min'],
            brake_params['max'])
        self.steer_ctrl = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.steer_filt = LowPassFilter(1.0, 0.0)
        self.prev_steer = 0.0

    def control(self, sample_time, exp_lin_vel, act_lin_vel, exp_ang_vel):
        lin_vel_err = exp_lin_vel - act_lin_vel

        # Compute the errors
        throttle = self.throttle_pid.step(lin_vel_err, sample_time)
        brake = self.brake_pid.step(-lin_vel_err, sample_time)

        # Sanitize the brake
        if lin_vel_err > 0:
            brake = 0.0

        # Compute the steering
        steering = self.steer_filt.filt(self.steer_ctrl.get_steering(exp_lin_vel, exp_ang_vel, act_lin_vel)) * 2.0

        # If there is too much of a difference in the steering angles
        # if (self.prev_steer - steering) > 1e-1:
        #     throttle = 0.2

        # Save the previous steering
        self.prev_steer = steering

        return throttle, brake, steering

    def reset(self):
        self.throttle_pid.reset()
        self.brake_pid.reset()
