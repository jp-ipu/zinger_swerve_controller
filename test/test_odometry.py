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

"""Tests for odometry integration in the steering controller."""

import math
import pytest
from typing import List

from zinger_swerve_controller.drive_module import DriveModule
from zinger_swerve_controller.geometry import Point
from zinger_swerve_controller.profile import SingleVariableLinearProfile
from zinger_swerve_controller.steering_controller import ModuleFollowsBodySteeringController


def create_test_drive_modules() -> List[DriveModule]:
    """Create a simple 2-wheel configuration for testing."""
    return [
        DriveModule(
            name="left",
            steering_link="steering_left",
            drive_link="drive_left",
            steering_axis_xy_position=Point(0.0, 0.1, 0.0),
            wheel_radius=0.05,
            wheel_width=0.03,
            steering_motor_maximum_velocity=1.0,
            steering_motor_minimum_acceleration=0.1,
            steering_motor_maximum_acceleration=1.0,
            steering_motor_maximum_jerk=100.0,
            drive_motor_maximum_velocity=1.0,
            drive_motor_minimum_acceleration=0.1,
            drive_motor_maximum_acceleration=1.0,
            drive_motor_maximum_jerk=100.0,
        ),
        DriveModule(
            name="right",
            steering_link="steering_right",
            drive_link="drive_right",
            steering_axis_xy_position=Point(0.0, -0.1, 0.0),
            wheel_radius=0.05,
            wheel_width=0.03,
            steering_motor_maximum_velocity=1.0,
            steering_motor_minimum_acceleration=0.1,
            steering_motor_maximum_acceleration=1.0,
            steering_motor_maximum_jerk=100.0,
            drive_motor_maximum_velocity=1.0,
            drive_motor_minimum_acceleration=0.1,
            drive_motor_maximum_acceleration=1.0,
            drive_motor_maximum_jerk=100.0,
        ),
    ]


def linear_profile_func(start: float, end: float):
    return SingleVariableLinearProfile(start, end)


def create_controller() -> ModuleFollowsBodySteeringController:
    """Create a steering controller for testing."""
    return ModuleFollowsBodySteeringController(
        drive_modules=create_test_drive_modules(),
        motion_profile_func=linear_profile_func,
        logger=lambda x: None,
    )


class TestArcIntegration:
    """Tests for the _integrate_odometry_arc method."""

    def test_pure_forward_motion(self):
        """Test forward motion with no rotation."""
        controller = create_controller()

        # Moving forward at 1 m/s for 1 second, starting at theta=0
        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=1.0,
            vy=0.0,
            omega=0.0,
            dt=1.0
        )

        assert math.isclose(dx, 1.0, rel_tol=1e-6)
        assert math.isclose(dy, 0.0, abs_tol=1e-6)
        assert math.isclose(new_theta, 0.0, abs_tol=1e-6)

    def test_pure_sideways_motion(self):
        """Test sideways motion with no rotation."""
        controller = create_controller()

        # Moving left at 1 m/s for 1 second, starting at theta=0
        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=0.0,
            vy=1.0,
            omega=0.0,
            dt=1.0
        )

        assert math.isclose(dx, 0.0, abs_tol=1e-6)
        assert math.isclose(dy, 1.0, rel_tol=1e-6)
        assert math.isclose(new_theta, 0.0, abs_tol=1e-6)

    def test_forward_motion_with_initial_orientation(self):
        """Test forward motion when robot is already rotated 90 degrees."""
        controller = create_controller()

        # Robot facing +Y direction (theta=pi/2), moving forward at 1 m/s
        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=math.pi / 2,
            vx=1.0,
            vy=0.0,
            omega=0.0,
            dt=1.0
        )

        # Forward in body frame should be +Y in world frame
        assert math.isclose(dx, 0.0, abs_tol=1e-6)
        assert math.isclose(dy, 1.0, rel_tol=1e-6)
        assert math.isclose(new_theta, math.pi / 2, rel_tol=1e-6)

    def test_pure_rotation(self):
        """Test pure rotation without translation."""
        controller = create_controller()

        # Rotating at 1 rad/s for 1 second
        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=0.0,
            vy=0.0,
            omega=1.0,
            dt=1.0
        )

        assert math.isclose(dx, 0.0, abs_tol=1e-6)
        assert math.isclose(dy, 0.0, abs_tol=1e-6)
        assert math.isclose(new_theta, 1.0, rel_tol=1e-6)

    def test_arc_motion_quarter_circle(self):
        """Test arc motion: driving forward while rotating 90 degrees."""
        controller = create_controller()

        # This should trace a quarter circle arc
        # With vx = omega * r, the robot moves in a circle of radius r
        # Let's use omega = pi/2 rad/s, dt = 1s, so d_theta = pi/2 (90 degrees)
        # And vx = 1 m/s, so radius = vx/omega = 2/pi meters
        omega = math.pi / 2
        vx = 1.0
        radius = vx / omega  # 2/pi meters

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=vx,
            vy=0.0,
            omega=omega,
            dt=1.0
        )

        # After a quarter circle starting at theta=0, moving forward:
        # The center of rotation is at (0, radius) from starting position
        # End position should be at (radius, radius) relative to start
        expected_dx = radius  # 2/pi
        expected_dy = radius  # 2/pi

        assert math.isclose(dx, expected_dx, rel_tol=1e-6)
        assert math.isclose(dy, expected_dy, rel_tol=1e-6)
        assert math.isclose(new_theta, math.pi / 2, rel_tol=1e-6)

    def test_arc_motion_full_circle(self):
        """Test arc motion: completing a full circle returns to start."""
        controller = create_controller()

        # Full circle: omega * dt = 2*pi
        omega = 2 * math.pi
        vx = 1.0

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=vx,
            vy=0.0,
            omega=omega,
            dt=1.0
        )

        # After a full circle, should return to origin
        assert math.isclose(dx, 0.0, abs_tol=1e-6)
        assert math.isclose(dy, 0.0, abs_tol=1e-6)
        assert math.isclose(new_theta, 2 * math.pi, rel_tol=1e-6)

    def test_arc_motion_half_circle(self):
        """Test arc motion: half circle should end up behind start."""
        controller = create_controller()

        # Half circle: omega * dt = pi
        omega = math.pi
        vx = 1.0
        radius = vx / omega  # 1/pi meters

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=vx,
            vy=0.0,
            omega=omega,
            dt=1.0
        )

        # After half circle starting at theta=0, moving forward:
        # Should end up at (0, 2*radius) - directly to the left of start
        expected_dx = 0.0
        expected_dy = 2 * radius  # 2/pi

        assert math.isclose(dx, expected_dx, abs_tol=1e-6)
        assert math.isclose(dy, expected_dy, rel_tol=1e-6)
        assert math.isclose(new_theta, math.pi, rel_tol=1e-6)

    def test_negative_rotation(self):
        """Test arc motion with negative (clockwise) rotation."""
        controller = create_controller()

        omega = -math.pi / 2  # Clockwise quarter turn
        vx = 1.0
        radius = vx / abs(omega)

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=vx,
            vy=0.0,
            omega=omega,
            dt=1.0
        )

        # Quarter circle clockwise: end up at (radius, -radius)
        expected_dx = radius
        expected_dy = -radius

        assert math.isclose(dx, expected_dx, rel_tol=1e-6)
        assert math.isclose(dy, expected_dy, rel_tol=1e-6)
        assert math.isclose(new_theta, -math.pi / 2, rel_tol=1e-6)

    def test_sideways_arc_motion(self):
        """Test arc motion while moving sideways."""
        controller = create_controller()

        # Moving sideways (vy) while rotating
        omega = math.pi / 2
        vy = 1.0
        radius = vy / omega

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=0.0,
            vy=vy,
            omega=omega,
            dt=1.0
        )

        # Moving left while rotating CCW - should curve forward
        # After quarter turn, end up at (-radius, radius)
        expected_dx = -radius
        expected_dy = radius

        assert math.isclose(dx, expected_dx, rel_tol=1e-6)
        assert math.isclose(dy, expected_dy, rel_tol=1e-6)

    def test_small_rotation_uses_midpoint(self):
        """Test that very small rotations don't cause numerical issues."""
        controller = create_controller()

        # Very small omega - should use midpoint approximation
        omega = 1e-8
        vx = 1.0

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=vx,
            vy=0.0,
            omega=omega,
            dt=1.0
        )

        # Should be approximately straight-line motion
        assert math.isclose(dx, 1.0, rel_tol=1e-6)
        assert math.isclose(dy, 0.0, abs_tol=1e-6)

    def test_combined_forward_and_sideways_with_rotation(self):
        """Test diagonal motion with rotation."""
        controller = create_controller()

        # Moving diagonally while rotating
        omega = math.pi / 4  # 45 degree turn
        vx = 1.0
        vy = 1.0

        dx, dy, new_theta = controller._integrate_odometry_arc(
            theta=0.0,
            vx=vx,
            vy=vy,
            omega=omega,
            dt=1.0
        )

        # The exact values depend on the arc integration
        # Just verify the function runs and produces reasonable output
        assert math.isfinite(dx)
        assert math.isfinite(dy)
        assert math.isclose(new_theta, math.pi / 4, rel_tol=1e-6)

        # The displacement should be roughly in the forward-left direction
        # but curved due to rotation
        assert dx > 0  # Still moving forward overall
        assert dy > 0  # Still moving left overall
