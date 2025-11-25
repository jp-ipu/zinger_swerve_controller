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

"""
Online trajectory generation using Ruckig for smooth velocity control.

This module provides real-time trajectory generation that respects acceleration
and jerk limits, allowing smooth transitions between velocity commands.
"""

from copy import copy
from dataclasses import dataclass
from typing import Callable, Optional

from ruckig import InputParameter, OutputParameter, Result, Ruckig, ControlInterface


@dataclass
class VelocityState:
    """Current or target velocity state for body motion."""
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class AccelerationState:
    """Current acceleration state for body motion."""
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


class RuckigVelocityController:
    """
    Online trajectory generator using Ruckig for smooth velocity control.

    This controller generates time-optimal trajectories that smoothly transition
    from the current velocity to a target velocity while respecting acceleration
    and jerk limits. It operates in velocity control mode, which is appropriate
    for cmd_vel style commands.

    The controller maintains internal state and should be called every control
    cycle (e.g., at 50Hz) to get the next velocity setpoint.

    Example usage:
        controller = RuckigVelocityController(
            control_cycle=0.02,  # 50Hz
            max_acceleration=[1.0, 1.0, 2.0],
            max_jerk=[10.0, 10.0, 20.0]
        )

        # When new cmd_vel arrives:
        controller.set_target_velocity(vx=1.0, vy=0.0, omega=0.5)

        # Every control cycle:
        velocity = controller.update()
        # Use velocity.linear_x, velocity.linear_y, velocity.angular_z
    """

    def __init__(
            self,
            control_cycle: float,
            max_acceleration: list[float],
            max_jerk: list[float],
            max_velocity: Optional[list[float]] = None,
            logger: Optional[Callable[[str], None]] = None):
        """
        Initialize the Ruckig velocity controller.

        Args:
            control_cycle: Control loop period in seconds (e.g., 0.02 for 50Hz)
            max_acceleration: Maximum acceleration for [linear_x, linear_y, angular_z]
            max_jerk: Maximum jerk for [linear_x, linear_y, angular_z]
            max_velocity: Optional maximum velocity limits for [linear_x, linear_y, angular_z]
            logger: Optional logging function
        """
        self.control_cycle = control_cycle
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.max_velocity = max_velocity
        self.logger = logger or (lambda x: None)

        # Create Ruckig instance for 3 DoFs: vx, vy, omega
        self.otg = Ruckig(3, control_cycle)
        self.inp = InputParameter(3)
        self.out = OutputParameter(3)

        # Use velocity control interface - we're controlling velocity, not position
        self.inp.control_interface = ControlInterface.Velocity

        # Initialize current state (at rest)
        self.inp.current_position = [0.0, 0.0, 0.0]  # Not used in velocity mode
        self.inp.current_velocity = [0.0, 0.0, 0.0]
        self.inp.current_acceleration = [0.0, 0.0, 0.0]

        # Set target (initially at rest)
        self.inp.target_velocity = [0.0, 0.0, 0.0]
        self.inp.target_acceleration = [0.0, 0.0, 0.0]

        # Set constraints
        self.inp.max_acceleration = list(max_acceleration)
        self.inp.max_jerk = list(max_jerk)

        # Optional velocity limits
        if max_velocity is not None:
            self.inp.max_velocity = list(max_velocity)

        # Track state
        self._target_velocity = VelocityState()
        self._current_velocity = VelocityState()
        self._current_acceleration = AccelerationState()
        self._is_at_target = True
        self._trajectory_duration = 0.0

    def set_target_velocity(
            self,
            linear_x: float = 0.0,
            linear_y: float = 0.0,
            angular_z: float = 0.0):
        """
        Set a new target velocity.

        The controller will smoothly transition to this velocity while respecting
        acceleration and jerk limits.

        Args:
            linear_x: Target linear velocity in x direction (m/s)
            linear_y: Target linear velocity in y direction (m/s)
            angular_z: Target angular velocity around z axis (rad/s)
        """
        self._target_velocity = VelocityState(linear_x, linear_y, angular_z)
        self.inp.target_velocity = [linear_x, linear_y, angular_z]
        self.inp.target_acceleration = [0.0, 0.0, 0.0]  # Come to rest at target velocity
        self._is_at_target = False

        self.logger(
            f'New target velocity: vx={linear_x:.3f}, vy={linear_y:.3f}, omega={angular_z:.3f}'
        )

    def update(self) -> VelocityState:
        """
        Compute the next velocity setpoint.

        This should be called every control cycle. It returns the velocity
        that should be commanded to achieve smooth motion toward the target.

        Returns:
            VelocityState with the commanded velocities for this cycle
        """
        if self._is_at_target:
            # Already at target, return target velocity
            return copy(self._target_velocity)

        # Run one step of the Ruckig algorithm
        result = self.otg.update(self.inp, self.out)

        if result == Result.Working:
            # Trajectory still in progress
            self._current_velocity = VelocityState(
                self.out.new_velocity[0],
                self.out.new_velocity[1],
                self.out.new_velocity[2]
            )
            self._current_acceleration = AccelerationState(
                self.out.new_acceleration[0],
                self.out.new_acceleration[1],
                self.out.new_acceleration[2]
            )

            # Pass output to input for next cycle
            self.out.pass_to_input(self.inp)

        elif result == Result.Finished:
            # Reached target
            self._current_velocity = copy(self._target_velocity)
            self._current_acceleration = AccelerationState(0.0, 0.0, 0.0)
            self._is_at_target = True
            self._trajectory_duration = self.out.trajectory.duration

            self.logger(
                f'Reached target velocity in {self._trajectory_duration:.3f}s'
            )

            # Update input state for next command
            self.inp.current_velocity = [
                self._current_velocity.linear_x,
                self._current_velocity.linear_y,
                self._current_velocity.angular_z
            ]
            self.inp.current_acceleration = [0.0, 0.0, 0.0]

        elif result == Result.Error:
            self.logger('Ruckig error - returning current velocity')
            # On error, maintain current velocity

        elif result == Result.ErrorInvalidInput:
            self.logger('Ruckig invalid input error')

        return copy(self._current_velocity)

    def update_current_state(
            self,
            velocity: VelocityState,
            acceleration: Optional[AccelerationState] = None):
        """
        Update the controller with measured state.

        Call this if you have feedback from the actual robot state to improve
        trajectory tracking. This is optional but recommended for better control.

        Args:
            velocity: Measured velocity state
            acceleration: Optional measured acceleration state
        """
        self._current_velocity = copy(velocity)
        self.inp.current_velocity = [velocity.linear_x, velocity.linear_y, velocity.angular_z]

        if acceleration is not None:
            self._current_acceleration = copy(acceleration)
            self.inp.current_acceleration = [
                acceleration.linear_x,
                acceleration.linear_y,
                acceleration.angular_z
            ]

    def stop(self):
        """Command a smooth stop (target velocity = 0)."""
        self.set_target_velocity(0.0, 0.0, 0.0)

    def emergency_stop(self):
        """Immediate stop - sets all velocities to zero without smooth trajectory."""
        self._target_velocity = VelocityState(0.0, 0.0, 0.0)
        self._current_velocity = VelocityState(0.0, 0.0, 0.0)
        self._current_acceleration = AccelerationState(0.0, 0.0, 0.0)
        self._is_at_target = True

        self.inp.current_velocity = [0.0, 0.0, 0.0]
        self.inp.current_acceleration = [0.0, 0.0, 0.0]
        self.inp.target_velocity = [0.0, 0.0, 0.0]
        self.inp.target_acceleration = [0.0, 0.0, 0.0]

        self.logger('Emergency stop executed')

    @property
    def current_velocity(self) -> VelocityState:
        """Get the current commanded velocity."""
        return copy(self._current_velocity)

    @property
    def target_velocity(self) -> VelocityState:
        """Get the target velocity."""
        return copy(self._target_velocity)

    @property
    def is_at_target(self) -> bool:
        """Check if the controller has reached the target velocity."""
        return self._is_at_target

    @property
    def trajectory_duration(self) -> float:
        """Get the duration of the last completed trajectory."""
        return self._trajectory_duration
