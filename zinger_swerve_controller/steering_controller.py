# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from typing import Callable, List

from .profile import TransientVariableProfile

# local
from .control import BodyMotionCommand, DriveModuleMotionCommand, InvalidMotionCommandException, MotionCommand
from .control_model import difference_between_angles, MultiWheelSteeringControlModel
from .control_profile import BodyMotionProfile, DriveModuleStateProfile
from .drive_module import DriveModule
from .states import BodyState, DriveModuleDesiredValues, DriveModuleMeasuredValues

class DriveModuleDesiredValuesProfilePoint():

    def __init__(self, time: float, drive_module_states: List[DriveModuleDesiredValues]):
        self.time_since_start_of_profile = time
        self.drive_module_states = drive_module_states

class ModuleFollowsBodySteeringController():

    def __init__(
            self,
            drive_modules: List[DriveModule],
            motion_profile_func: Callable[[float, float], TransientVariableProfile],
            logger: Callable[[str], None]):
        # Get the geometry for the robot
        self.modules = drive_modules
        self.motion_profile_func = motion_profile_func
        self.logger = logger

        # Use a multi-wheel control model that supports 2, 3, 4, or more wheels
        self.control_model = MultiWheelSteeringControlModel(self.modules)

        # Store the current (estimated) state of the body
        self.body_state: BodyState = BodyState(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        # Store the current (measured) state of the drive modules
        self.module_states: List[DriveModuleMeasuredValues] = [
            DriveModuleMeasuredValues(
                drive_module.name,
                drive_module.steering_axis_xy_position.x,
                drive_module.steering_axis_xy_position.y,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ) for drive_module in drive_modules
        ]

        self.previous_module_states: List[DriveModuleMeasuredValues] = [
            DriveModuleMeasuredValues(
                drive_module.name,
                drive_module.steering_axis_xy_position.x,
                drive_module.steering_axis_xy_position.y,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ) for drive_module in drive_modules
        ]

        # Profiles
        self.body_profile: BodyMotionProfile = None
        self.module_profile_from_command: DriveModuleStateProfile = None

         # Keep track of our position in time so that we can figure out where on the current
        # profile we should be
        self.current_time_in_seconds = 0.0
        self.profile_was_started_at_time_in_seconds = 0.0
        self.last_state_update_time = 0.0
        self.min_time_for_profile: float = 0.0

        # flags
        self.is_executing_body_profile: bool = False
        self.is_executing_module_profile: bool = False

    def body_state_at_current_time(self) -> BodyState:
        return self.body_state

    def drive_module_states_at_current_time(self) -> List[DriveModuleMeasuredValues]:
        return self.module_states

    def _select_best_steering_state(
        self,
        module: 'DriveModule',
        current_steering_angle: float,
        current_velocity: float,
        forward_state: DriveModuleDesiredValues,
        reverse_state: DriveModuleDesiredValues
    ) -> DriveModuleDesiredValues:
        """
        Select the best steering state considering angle limits and minimizing rotation.

        Args:
            module: The drive module (contains steering limits)
            current_steering_angle: Current steering angle in radians
            current_velocity: Current drive velocity in m/s
            forward_state: Forward direction option
            reverse_state: Reverse direction option (angle + 180°, negative velocity)

        Returns:
            The best DriveModuleDesiredValues to use
        """
        # Check which states are reachable within steering limits
        forward_reachable = module.is_steering_angle_reachable(forward_state.steering_angle_in_radians)
        reverse_reachable = module.is_steering_angle_reachable(reverse_state.steering_angle_in_radians)

        # If neither is reachable, clamp to the nearest limit
        if not forward_reachable and not reverse_reachable:
            # Find which state is closer to the limits
            forward_angle = forward_state.steering_angle_in_radians
            reverse_angle = reverse_state.steering_angle_in_radians

            if not math.isinf(forward_angle):
                # Clamp forward angle to limits
                clamped_forward = max(module.steering_angle_min,
                                     min(module.steering_angle_max, forward_angle))
                forward_distance = abs(difference_between_angles(current_steering_angle, clamped_forward))
            else:
                forward_distance = 0  # No steering change needed

            if not math.isinf(reverse_angle):
                clamped_reverse = max(module.steering_angle_min,
                                     min(module.steering_angle_max, reverse_angle))
                reverse_distance = abs(difference_between_angles(current_steering_angle, clamped_reverse))
            else:
                reverse_distance = 0

            # Return the state with clamped angle that's closest
            if forward_distance <= reverse_distance:
                if not math.isinf(forward_angle):
                    clamped_angle = max(module.steering_angle_min,
                                       min(module.steering_angle_max, forward_angle))
                    return DriveModuleDesiredValues(
                        forward_state.name,
                        clamped_angle,
                        forward_state.drive_velocity_in_meters_per_second
                    )
                return forward_state
            else:
                if not math.isinf(reverse_angle):
                    clamped_angle = max(module.steering_angle_min,
                                       min(module.steering_angle_max, reverse_angle))
                    return DriveModuleDesiredValues(
                        reverse_state.name,
                        clamped_angle,
                        reverse_state.drive_velocity_in_meters_per_second
                    )
                return reverse_state

        # If only one is reachable, use that one
        if forward_reachable and not reverse_reachable:
            return forward_state
        if reverse_reachable and not forward_reachable:
            return reverse_state

        # Both are reachable - use original logic to pick the best one
        forward_rotation_diff = difference_between_angles(
            current_steering_angle, forward_state.steering_angle_in_radians)
        reverse_rotation_diff = difference_between_angles(
            current_steering_angle, reverse_state.steering_angle_in_radians)

        forward_velocity_diff = forward_state.drive_velocity_in_meters_per_second - current_velocity
        reverse_velocity_diff = reverse_state.drive_velocity_in_meters_per_second - current_velocity

        # Pick the state with smallest rotation, using velocity as tiebreaker
        if abs(forward_rotation_diff) <= abs(reverse_rotation_diff):
            if abs(forward_velocity_diff) <= abs(reverse_velocity_diff):
                return forward_state
            else:
                if math.isclose(abs(forward_rotation_diff), abs(reverse_rotation_diff),
                               rel_tol=1e-7, abs_tol=1e-7):
                    return reverse_state
                else:
                    return forward_state
        else:
            if abs(reverse_velocity_diff) <= abs(forward_velocity_diff):
                return reverse_state
            else:
                if math.isclose(abs(forward_rotation_diff), abs(reverse_rotation_diff),
                               rel_tol=1e-7, abs_tol=1e-7):
                    return forward_state
                else:
                    return reverse_state

    def drive_module_state_at_profile_time(self, time_fraction: float) -> List[DriveModuleDesiredValues]:
        result: List[DriveModuleDesiredValues] = []
        if self.is_executing_body_profile:
            body_state = self.body_profile.body_motion_at(time_fraction)
            drive_module_desired_values = self.control_model.state_of_wheel_modules_from_body_motion(body_state)
            for i in range(len(self.modules)):
                current_state_for_module = self.module_states[i]
                current_steering_angle = current_state_for_module.orientation_in_body_coordinates.z
                current_velocity = current_state_for_module.drive_velocity_in_module_coordinates.x

                states_for_module = drive_module_desired_values[i]
                forward_state = states_for_module[0]
                reverse_state = states_for_module[1]

                selected_state = self._select_best_steering_state(
                    self.modules[i],
                    current_steering_angle,
                    current_velocity,
                    forward_state,
                    reverse_state
                )
                result.append(selected_state)
        else:
            for drive_module in self.modules:
                state = self.module_profile_from_command.value_for_module_at(drive_module.name, time_fraction)
                result.append(DriveModuleDesiredValues(
                    state.name,
                    state.orientation_in_body_coordinates.z,
                    state.drive_velocity_in_module_coordinates.x
                ))

        return result

    # Returns the state of the drive modules to required to match the current profile at the given
    # time.
    def drive_module_state_at_future_time(self, future_time_in_seconds:float) -> List[DriveModuleDesiredValues]:
        if self.body_profile is None and self.module_profile_from_command is None:
            return []

        time_from_start_of_profile = future_time_in_seconds - self.profile_was_started_at_time_in_seconds
        # self.logger(
        #     'Determining profile values at {}'.format(time_from_start_of_profile)
        # )

        profile_time = self.body_profile.time_span() if self.is_executing_body_profile else self.module_profile_from_command.time_span()
        time_fraction = time_from_start_of_profile / profile_time

        result: List[DriveModuleDesiredValues] = self.drive_module_state_at_profile_time(time_fraction)
        return result

    def drive_module_profile_points_from_now_till_end(self, starting_time: float) -> List[DriveModuleDesiredValuesProfilePoint]:
        if self.body_profile is None and self.module_profile_from_command is None:
            return []

        # for now distribute the points equally. But really what we should be doing is putting more points in
        # places where the second or third derivatives change sign

        time_from_start_of_profile = starting_time - self.profile_was_started_at_time_in_seconds

        profile_time = self.body_profile.time_span() if self.is_executing_body_profile else self.module_profile_from_command.time_span()

        division_count = 10
        time_fraction_start = (time_from_start_of_profile / profile_time)
        time_fraction_end = 1.0 * division_count

        # Take time steps of 1/100 of the total profile time. Find the next time step we should take and
        # then find the number of steps we have left to take in the current
        next_time_step = int(self.round_up(time_fraction_start, 1.0 / division_count) * division_count)

        result: List[DriveModuleDesiredValuesProfilePoint] = []
        for step in range(next_time_step, time_fraction_end, 1):
            time_fraction = (float(step)) / division_count
            time = profile_time * time_fraction
            states = self.drive_module_state_at_profile_time(time_fraction)

            point = DriveModuleDesiredValuesProfilePoint(time, states)
            result.append(point)

        return result

    # Updates the currently stored desired body state. On the next time tick the
    # drive module trajectory will be updated to match the new desired end state.
    def on_desired_state_update(self, desired_motion: MotionCommand):
        if isinstance(desired_motion, BodyMotionCommand):
            trajectory = BodyMotionProfile(
                self.body_state,
                desired_motion.to_body_state(self.control_model),
                desired_motion.time_for_motion(),
                self.motion_profile_func)
            self.body_profile = trajectory

            self.is_executing_body_profile = True
            self.is_executing_module_profile = False

            self.logger(
                'Starting body motion profile with starting state [[x:{}, y:{}, o:{}],[vx:{}, vy:{}, vo:{}]] and desired end state [vx:{}, vy:{},vo:{}]'.format(
                    self.body_state.position_in_world_coordinates.x,
                    self.body_state.position_in_world_coordinates.y,
                    self.body_state.orientation_in_world_coordinates.z,
                    self.body_state.motion_in_body_coordinates.linear_velocity.x,
                    self.body_state.motion_in_body_coordinates.linear_velocity.y,
                    self.body_state.motion_in_body_coordinates.angular_velocity.z,
                    desired_motion.linear_velocity.x,
                    desired_motion.linear_velocity.y,
                    desired_motion.angular_velocity.z,)
            )
        else:
            if isinstance(desired_motion, DriveModuleMotionCommand):
                trajectory = DriveModuleStateProfile(self.modules, desired_motion.time_for_motion(), self.motion_profile_func)
                trajectory.set_current_state(self.module_states)
                trajectory.set_desired_end_state(desired_motion.to_drive_module_state(self.control_model)[0])
                self.module_profile_from_command = trajectory

                self.is_executing_body_profile = False
                self.is_executing_module_profile = True

                self.logger(
                    'Starting module motion profile with starting state {} and desired end state {}'.format(self.body_state, desired_motion)
                )
            else:
                raise InvalidMotionCommandException()

        self.profile_was_started_at_time_in_seconds = self.current_time_in_seconds
        self.min_time_for_profile = desired_motion.time_for_motion()

    # Updates the currently stored drive module state
    def on_state_update(
            self,
            current_module_states: List[DriveModuleMeasuredValues],
            measurement_time_in_seconds: float = None):
        """
        Update odometry based on new wheel state measurements.

        Args:
            current_module_states: Current measured state of each drive module
            measurement_time_in_seconds: Timestamp when the measurement was taken.
                If None, uses current_time_in_seconds (less accurate due to latency).
                For best accuracy, pass the timestamp from the sensor message header.
        """
        if current_module_states is None:
            raise TypeError()

        if len(current_module_states) != len(self.modules):
            raise ValueError()

        # Use measurement timestamp if provided, otherwise fall back to current time
        if measurement_time_in_seconds is None:
            measurement_time_in_seconds = self.current_time_in_seconds

        self.previous_module_states = self.module_states
        self.module_states = current_module_states

        # Calculate the current body motion from wheel states (forward kinematics)
        body_motion = self.control_model.body_motion_from_wheel_module_states(self.module_states)

        # Compute dt using measurement timestamps for accurate integration
        # This accounts for sensor latency and message delivery delays
        time_step_in_seconds = measurement_time_in_seconds - self.last_state_update_time

        # Use trapezoidal integration for velocities (average of old and new)
        avg_vx = 0.5 * (self.body_state.motion_in_body_coordinates.linear_velocity.x + body_motion.linear_velocity.x)
        avg_vy = 0.5 * (self.body_state.motion_in_body_coordinates.linear_velocity.y + body_motion.linear_velocity.y)
        avg_omega = 0.5 * (self.body_state.motion_in_body_coordinates.angular_velocity.z + body_motion.angular_velocity.z)

        # Calculate position change using arc-based integration
        # This properly handles the case where the robot is rotating while translating
        global_dx, global_dy, new_orientation = self._integrate_odometry_arc(
            self.body_state.orientation_in_world_coordinates.z,
            avg_vx,
            avg_vy,
            avg_omega,
            time_step_in_seconds
        )

        new_x = self.body_state.position_in_world_coordinates.x + global_dx
        new_y = self.body_state.position_in_world_coordinates.y + global_dy

        # Compute acceleration and jerk from velocity changes (for reference/debugging)
        # Note: These are numerical derivatives and can be noisy
        local_x_acceleration = 0.0
        local_y_acceleration = 0.0
        orientation_acceleration = 0.0
        local_x_jerk = 0.0
        local_y_jerk = 0.0
        orientation_jerk = 0.0

        if not math.isclose(time_step_in_seconds, 0.0, abs_tol=1e-4, rel_tol=1e-4):
            local_x_acceleration = (body_motion.linear_velocity.x - self.body_state.motion_in_body_coordinates.linear_velocity.x) / time_step_in_seconds
            local_y_acceleration = (body_motion.linear_velocity.y - self.body_state.motion_in_body_coordinates.linear_velocity.y) / time_step_in_seconds
            orientation_acceleration = (body_motion.angular_velocity.z - self.body_state.motion_in_body_coordinates.angular_velocity.z) / time_step_in_seconds

            local_x_jerk = (local_x_acceleration - self.body_state.motion_in_body_coordinates.linear_acceleration.x) / time_step_in_seconds
            local_y_jerk = (local_y_acceleration - self.body_state.motion_in_body_coordinates.linear_acceleration.y) / time_step_in_seconds
            orientation_jerk = (orientation_acceleration - self.body_state.motion_in_body_coordinates.angular_acceleration.z) / time_step_in_seconds

        self.body_state = BodyState(
            new_x,
            new_y,
            new_orientation,
            body_motion.linear_velocity.x,
            body_motion.linear_velocity.y,
            body_motion.angular_velocity.z,
            local_x_acceleration,
            local_y_acceleration,
            orientation_acceleration,
            local_x_jerk,
            local_y_jerk,
            orientation_jerk
        )

        self.last_state_update_time = measurement_time_in_seconds

    def _integrate_odometry_arc(
            self,
            theta: float,
            vx: float,
            vy: float,
            omega: float,
            dt: float) -> tuple:
        """
        Integrate odometry using exact arc-based equations.

        When the robot rotates while translating, it follows an arc rather than
        a straight line. This method computes the exact displacement for constant
        velocity and angular velocity over the time step.

        For pure translation (omega ≈ 0), this reduces to simple Euler integration.
        For rotation with translation, it uses the closed-form arc solution.

        Args:
            theta: Current orientation in world frame (radians)
            vx: Body-frame x velocity (m/s)
            vy: Body-frame y velocity (m/s)
            omega: Angular velocity (rad/s)
            dt: Time step (seconds)

        Returns:
            (dx_global, dy_global, new_theta): Displacement in world frame and new orientation
        """
        d_theta = omega * dt
        new_theta = theta + d_theta

        # Threshold for "effectively zero" rotation
        # Below this, use linear approximation to avoid division by near-zero
        OMEGA_THRESHOLD = 1e-6

        if abs(omega) < OMEGA_THRESHOLD:
            # Pure translation (or negligible rotation)
            # Use midpoint orientation for better accuracy
            mid_theta = theta + 0.5 * d_theta
            cos_mid = math.cos(mid_theta)
            sin_mid = math.sin(mid_theta)

            dx_local = vx * dt
            dy_local = vy * dt

            dx_global = dx_local * cos_mid - dy_local * sin_mid
            dy_global = dx_local * sin_mid + dy_local * cos_mid
        else:
            # Arc motion: exact integration for constant v and omega
            # The robot moves along an arc while rotating
            #
            # Derivation:
            # x(t) = ∫ vx*cos(θ+ωt) - vy*sin(θ+ωt) dt
            # y(t) = ∫ vx*sin(θ+ωt) + vy*cos(θ+ωt) dt
            #
            # Solving these integrals gives:
            # Δx = (vx*(sin(θ+ωdt) - sin(θ)) + vy*(cos(θ+ωdt) - cos(θ))) / ω
            # Δy = (vx*(-cos(θ+ωdt) + cos(θ)) + vy*(sin(θ+ωdt) - sin(θ))) / ω
            #
            # Simplified using sin/cos difference identities:
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            sin_new = math.sin(new_theta)
            cos_new = math.cos(new_theta)

            dx_global = (vx * (sin_new - sin_theta) + vy * (cos_new - cos_theta)) / omega
            dy_global = (vx * (cos_theta - cos_new) + vy * (sin_new - sin_theta)) / omega

        return dx_global, dy_global, new_theta

    # On clock tick, determine if we need to recalculate the trajectories for the drive modules
    def on_tick(self, current_time_in_seconds: float):
        self.current_time_in_seconds = current_time_in_seconds

    def round_down(self, num: float, to: float) -> float:
        if num < 0:
            return -self.round_up(-num, to)
        mod = math.fmod(num, to)

        return num if math.isclose(mod, to) else num - mod

    def round_up(self, num: float, to: float) -> float:
        if num < 0:
            return -self.round_down(-num, to)

        down = self.round_down(num, to)

        return num if num == down else down + to
