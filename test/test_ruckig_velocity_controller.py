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
import pytest

from zinger_swerve_controller.ruckig_velocity_controller import (
    RuckigVelocityController,
    VelocityState,
    AccelerationState,
)


def create_controller(
    control_cycle: float = 0.02,
    max_acceleration: list = None,
    max_jerk: list = None,
    max_velocity: list = None
) -> RuckigVelocityController:
    """Helper to create a controller with reasonable defaults."""
    if max_acceleration is None:
        max_acceleration = [1.0, 1.0, 2.0]
    if max_jerk is None:
        max_jerk = [10.0, 10.0, 20.0]

    return RuckigVelocityController(
        control_cycle=control_cycle,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
        max_velocity=max_velocity
    )


def test_controller_starts_at_rest():
    """Test that the controller starts with zero velocity."""
    controller = create_controller()

    velocity = controller.current_velocity
    assert velocity.linear_x == 0.0
    assert velocity.linear_y == 0.0
    assert velocity.angular_z == 0.0
    assert controller.is_at_target


def test_controller_reaches_target_velocity():
    """Test that the controller eventually reaches the target velocity."""
    controller = create_controller()

    controller.set_target_velocity(linear_x=1.0, linear_y=0.0, angular_z=0.0)

    # Run enough cycles to reach target
    for _ in range(1000):
        velocity = controller.update()
        if controller.is_at_target:
            break

    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, 1.0, rel_tol=1e-3)
    assert math.isclose(velocity.linear_y, 0.0, abs_tol=1e-6)
    assert math.isclose(velocity.angular_z, 0.0, abs_tol=1e-6)


def test_controller_respects_acceleration_limits():
    """Test that acceleration never exceeds the limit during transitions."""
    max_accel = [2.0, 2.0, 4.0]
    controller = create_controller(
        control_cycle=0.01,
        max_acceleration=max_accel,
        max_jerk=[100.0, 100.0, 200.0]  # High jerk to let acceleration limits dominate
    )

    controller.set_target_velocity(linear_x=10.0, linear_y=0.0, angular_z=0.0)

    prev_velocity = VelocityState(0.0, 0.0, 0.0)
    dt = 0.01

    for _ in range(500):
        velocity = controller.update()

        # Calculate approximate acceleration
        accel_x = abs(velocity.linear_x - prev_velocity.linear_x) / dt

        # Allow some tolerance for numerical precision
        assert accel_x <= max_accel[0] + 0.1, \
            f"Acceleration {accel_x} exceeds limit {max_accel[0]}"

        prev_velocity = velocity
        if controller.is_at_target:
            break


def test_controller_smooth_velocity_change():
    """Test that velocity changes smoothly (no discontinuities)."""
    controller = create_controller(control_cycle=0.01)

    controller.set_target_velocity(linear_x=1.0, linear_y=0.5, angular_z=0.2)

    velocities = []
    for _ in range(200):
        velocity = controller.update()
        velocities.append(velocity)
        if controller.is_at_target:
            break

    # Check that velocity changes are smooth (no large jumps)
    for i in range(1, len(velocities)):
        dx = abs(velocities[i].linear_x - velocities[i-1].linear_x)
        dy = abs(velocities[i].linear_y - velocities[i-1].linear_y)
        dz = abs(velocities[i].angular_z - velocities[i-1].angular_z)

        # Maximum change per cycle should be limited by acceleration
        # With dt=0.01 and max_accel=1.0, max change is 0.01 m/s per cycle
        assert dx < 0.1, f"Large velocity jump in x: {dx}"
        assert dy < 0.1, f"Large velocity jump in y: {dy}"
        assert dz < 0.2, f"Large velocity jump in omega: {dz}"


def test_controller_handles_direction_change():
    """Test smooth transition when reversing direction."""
    controller = create_controller()

    # First move forward
    controller.set_target_velocity(linear_x=1.0, linear_y=0.0, angular_z=0.0)
    for _ in range(200):
        controller.update()
        if controller.is_at_target:
            break

    assert controller.is_at_target

    # Now reverse
    controller.set_target_velocity(linear_x=-1.0, linear_y=0.0, angular_z=0.0)

    for _ in range(500):
        velocity = controller.update()
        if controller.is_at_target:
            break

    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, -1.0, rel_tol=1e-3)


def test_controller_stop():
    """Test the stop command brings velocity to zero."""
    controller = create_controller()

    # Set velocity
    controller.set_target_velocity(linear_x=1.0, linear_y=0.5, angular_z=0.3)
    for _ in range(200):
        controller.update()
        if controller.is_at_target:
            break

    # Now stop
    controller.stop()

    for _ in range(500):
        velocity = controller.update()
        if controller.is_at_target:
            break

    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, 0.0, abs_tol=1e-6)
    assert math.isclose(velocity.linear_y, 0.0, abs_tol=1e-6)
    assert math.isclose(velocity.angular_z, 0.0, abs_tol=1e-6)


def test_controller_emergency_stop():
    """Test emergency stop immediately zeros velocity."""
    controller = create_controller()

    # Set velocity
    controller.set_target_velocity(linear_x=1.0, linear_y=0.5, angular_z=0.3)
    for _ in range(50):  # Only partial transition
        controller.update()

    # Emergency stop
    controller.emergency_stop()

    assert controller.is_at_target
    velocity = controller.current_velocity
    assert velocity.linear_x == 0.0
    assert velocity.linear_y == 0.0
    assert velocity.angular_z == 0.0


def test_controller_handles_mid_trajectory_command_change():
    """Test that controller smoothly handles command changes mid-trajectory."""
    controller = create_controller()

    # Start moving forward
    controller.set_target_velocity(linear_x=2.0, linear_y=0.0, angular_z=0.0)

    # Run for a bit
    for _ in range(50):
        velocity = controller.update()

    # Change command mid-trajectory
    mid_velocity = velocity.linear_x
    assert mid_velocity > 0 and mid_velocity < 2.0, "Should be mid-trajectory"

    # Change to a different target
    controller.set_target_velocity(linear_x=0.5, linear_y=1.0, angular_z=0.0)

    # Continue and check smooth transition
    prev_velocity = velocity
    for _ in range(500):
        velocity = controller.update()

        # Check no large discontinuities
        dx = abs(velocity.linear_x - prev_velocity.linear_x)
        dy = abs(velocity.linear_y - prev_velocity.linear_y)
        assert dx < 0.1, f"Discontinuity in x: {dx}"
        assert dy < 0.1, f"Discontinuity in y: {dy}"

        prev_velocity = velocity
        if controller.is_at_target:
            break

    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, 0.5, rel_tol=1e-3)
    assert math.isclose(velocity.linear_y, 1.0, rel_tol=1e-3)


def test_controller_update_current_state():
    """Test that updating current state from feedback works."""
    controller = create_controller()

    # Simulate external state update (e.g., from robot feedback)
    controller.update_current_state(
        velocity=VelocityState(0.5, 0.0, 0.0),
        acceleration=AccelerationState(0.1, 0.0, 0.0)
    )

    # Set a target
    controller.set_target_velocity(linear_x=1.0, linear_y=0.0, angular_z=0.0)

    # First update should start from the updated state
    velocity = controller.update()

    # Should be close to 0.5 (the starting velocity we set)
    assert abs(velocity.linear_x - 0.5) < 0.2


def test_controller_handles_high_velocity_targets():
    """Test that the controller handles high velocity targets smoothly.

    Note: In velocity control mode, Ruckig doesn't clamp the target velocity -
    it will smoothly accelerate toward whatever target is set. The max_velocity
    parameter affects the trajectory shape but not the final target.
    """
    controller = create_controller()

    # Set a high velocity target
    controller.set_target_velocity(linear_x=5.0, linear_y=0.0, angular_z=0.0)

    # Run to completion
    for _ in range(1000):
        velocity = controller.update()
        if controller.is_at_target:
            break

    # Should eventually reach the target
    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, 5.0, rel_tol=1e-3)


def test_controller_trajectory_duration():
    """Test that trajectory duration is tracked."""
    controller = create_controller()

    controller.set_target_velocity(linear_x=1.0, linear_y=0.0, angular_z=0.0)

    for _ in range(500):
        controller.update()
        if controller.is_at_target:
            break

    # Should have a positive duration
    assert controller.trajectory_duration > 0


def test_controller_angular_velocity():
    """Test pure rotation command."""
    controller = create_controller()

    controller.set_target_velocity(linear_x=0.0, linear_y=0.0, angular_z=1.0)

    for _ in range(500):
        velocity = controller.update()
        if controller.is_at_target:
            break

    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, 0.0, abs_tol=1e-6)
    assert math.isclose(velocity.linear_y, 0.0, abs_tol=1e-6)
    assert math.isclose(velocity.angular_z, 1.0, rel_tol=1e-3)


def test_controller_combined_motion():
    """Test combined linear and angular motion."""
    controller = create_controller()

    controller.set_target_velocity(linear_x=0.5, linear_y=0.3, angular_z=0.2)

    for _ in range(500):
        velocity = controller.update()
        if controller.is_at_target:
            break

    assert controller.is_at_target
    assert math.isclose(velocity.linear_x, 0.5, rel_tol=1e-3)
    assert math.isclose(velocity.linear_y, 0.3, rel_tol=1e-3)
    assert math.isclose(velocity.angular_z, 0.2, rel_tol=1e-3)
