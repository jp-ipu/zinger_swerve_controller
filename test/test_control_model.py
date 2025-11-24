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
from typing import Mapping, List, Tuple

# locals
from zinger_swerve_controller.control_model import (
    difference_between_angles,
    normalize_angle,
    MultiWheelSteeringControlModel,
    SimpleFourWheelSteeringControlModel,  # backward-compatible alias
)
from zinger_swerve_controller.drive_module import DriveModule
from zinger_swerve_controller.geometry import Point
from zinger_swerve_controller.states import BodyMotion, BodyState, DriveModuleMeasuredValues


def create_drive_module(
    name: str,
    x: float,
    y: float,
    wheel_radius: float = 0.1,
    wheel_width: float = 0.05,
    steering_max_velocity: float = 1.0,
    steering_min_acceleration: float = 0.1,
    steering_max_acceleration: float = 1.0,
    drive_max_velocity: float = 1.0,
    drive_min_acceleration: float = 0.1,
    drive_max_acceleration: float = 1.0) -> DriveModule:
    """Helper to create a single drive module at a given position."""
    return DriveModule(
        name=name,
        steering_link=f"steering_link_{name}",
        drive_link=f"drive_link_{name}",
        steering_axis_xy_position=Point(x, y, 0.0),
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_motor_maximum_velocity=steering_max_velocity,
        steering_motor_minimum_acceleration=steering_min_acceleration,
        steering_motor_maximum_acceleration=steering_max_acceleration,
        drive_motor_maximum_velocity=drive_max_velocity,
        drive_motor_minimum_acceleration=drive_min_acceleration,
        drive_motor_maximum_acceleration=drive_max_acceleration
    )


def create_two_wheel_drive_modules(
    length: float = 1.0,
    wheel_radius: float = 0.1,
    wheel_width: float = 0.05,
    steering_max_velocity: float = 1.0,
    steering_min_acceleration: float = 0.1,
    steering_max_acceleration: float = 1.0,
    drive_max_velocity: float = 1.0,
    drive_min_acceleration: float = 0.1,
    drive_max_acceleration: float = 1.0) -> List[DriveModule]:
    """Create a 2-wheel configuration with front and rear modules on the centerline."""
    result: List[DriveModule] = []

    front_drive = create_drive_module(
        name="front",
        x=0.5 * length,
        y=0.0,
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_max_velocity=steering_max_velocity,
        steering_min_acceleration=steering_min_acceleration,
        steering_max_acceleration=steering_max_acceleration,
        drive_max_velocity=drive_max_velocity,
        drive_min_acceleration=drive_min_acceleration,
        drive_max_acceleration=drive_max_acceleration
    )
    result.append(front_drive)

    rear_drive = create_drive_module(
        name="rear",
        x=-0.5 * length,
        y=0.0,
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_max_velocity=steering_max_velocity,
        steering_min_acceleration=steering_min_acceleration,
        steering_max_acceleration=steering_max_acceleration,
        drive_max_velocity=drive_max_velocity,
        drive_min_acceleration=drive_min_acceleration,
        drive_max_acceleration=drive_max_acceleration
    )
    result.append(rear_drive)

    return result


def create_drive_modules(
    length: float = 1.0,
    width: float = 1.0,
    wheel_radius: float = 0.1,
    wheel_width: float = 0.05,
    steering_max_velocity: float = 1.0,
    steering_min_acceleration: float = 0.1,
    steering_max_acceleration: float = 1.0,
    drive_max_velocity: float = 1.0,
    drive_min_acceleration: float = 0.1,
    drive_max_acceleration: float = 1.0) -> List[DriveModule]:
    """Create a 4-wheel configuration (original behavior)."""
    result: List[DriveModule] = []

    right_front_drive = DriveModule(
        name="module_1",
        steering_link="steering_link_1",
        drive_link="drive_link_1",
        steering_axis_xy_position=Point(0.5 * length, -0.5 * width, 0.0),
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_motor_maximum_velocity=steering_max_velocity,
        steering_motor_minimum_acceleration=steering_min_acceleration,
        steering_motor_maximum_acceleration=steering_max_acceleration,
        drive_motor_maximum_velocity=drive_max_velocity,
        drive_motor_minimum_acceleration=drive_min_acceleration,
        drive_motor_maximum_acceleration=drive_max_acceleration
    )
    result.append(right_front_drive)

    left_front_drive = DriveModule(
        name="module_2",
        steering_link="steering_link_2",
        drive_link="drive_link_2",
        steering_axis_xy_position=Point(0.5 * length, 0.5 * width, 0.0),
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_motor_maximum_velocity=steering_max_velocity,
        steering_motor_minimum_acceleration=steering_min_acceleration,
        steering_motor_maximum_acceleration=steering_max_acceleration,
        drive_motor_maximum_velocity=drive_max_velocity,
        drive_motor_minimum_acceleration=drive_min_acceleration,
        drive_motor_maximum_acceleration=drive_max_acceleration
    )
    result.append(left_front_drive)

    left_rear_drive = DriveModule(
        name="module_3",
        steering_link="steering_link_3",
        drive_link="drive_link_3",
        steering_axis_xy_position=Point(-0.5 * length, 0.5 * width, 0.0),
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_motor_maximum_velocity=steering_max_velocity,
        steering_motor_minimum_acceleration=steering_min_acceleration,
        steering_motor_maximum_acceleration=steering_max_acceleration,
        drive_motor_maximum_velocity=drive_max_velocity,
        drive_motor_minimum_acceleration=drive_min_acceleration,
        drive_motor_maximum_acceleration=drive_max_acceleration
    )
    result.append(left_rear_drive)

    right_rear_drive = DriveModule(
        name="module_4",
        steering_link="steering_link_4",
        drive_link="drive_link_4",
        steering_axis_xy_position=Point(-0.5 * length, -0.5 * width, 0.0),
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        steering_motor_maximum_velocity=steering_max_velocity,
        steering_motor_minimum_acceleration=steering_min_acceleration,
        steering_motor_maximum_acceleration=steering_max_acceleration,
        drive_motor_maximum_velocity=drive_max_velocity,
        drive_motor_minimum_acceleration=drive_min_acceleration,
        drive_motor_maximum_acceleration=drive_max_acceleration
    )
    result.append(right_rear_drive)

    return result

# normalize_angle

def test_should_normalize_angle():
    assert math.isclose(normalize_angle(0.0), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(0.5 * math.pi), 0.5 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(math.pi), math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(1.5 * math.pi), -0.5 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(2 * math.pi), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(2.5 * math.pi), 0.5 * math.pi, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(normalize_angle(4.0 * math.pi), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(5.0 * math.pi), math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(100 * math.pi), 0.0, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(normalize_angle(-math.pi + 1e-6), -math.pi + 1e-6, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(normalize_angle(-1.5 * math.pi), 0.5 * math.pi, rel_tol=1e-6, abs_tol=1e-6)

def test_should_calculate_difference_between_angles():
    assert math.isclose(difference_between_angles(0.0, 0.0), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(difference_between_angles(0.0, 2 * math.pi), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(difference_between_angles(2 * math.pi, 0.0), 0.0, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(difference_between_angles(math.pi, -math.pi), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(difference_between_angles(-math.pi, math.pi), 0.0, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(difference_between_angles(0.5 * math.pi, -0.5 * math.pi), -math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(difference_between_angles(0.5 * math.pi, -math.pi), 0.5 * math.pi, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(difference_between_angles(179.0/360.0 * 2 * math.pi, 181.0/360.0 * 2 * math.pi), 2.0 / 360.0 * 2 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(difference_between_angles(1.0/360.0 * 2 * math.pi, 359.0/360.0 * 2 * math.pi), -2.0 / 360.0 * 2 * math.pi, rel_tol=1e-6, abs_tol=1e-6)

# body_motion_from_wheel_module_states

def test_should_show_forward_movement_when_modules_are_pointing_forward_with_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(0),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, 1.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_no_movement_when_modules_are_pointing_forward_without_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_reverse_movement_when_modules_are_pointing_forward_with_negative_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(0),
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, -1.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_reverse_movement_when_modules_are_pointing_backwards_with_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(180),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, -1.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_left_sideways_movement_when_modules_are_pointing_left_with_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(90),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 1.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_right_sideways_movement_when_modules_are_pointing_left_with_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(270),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, -1.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_left_diagonal_movement_when_modules_are_pointing_left_diagonal_with_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(45),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, math.sqrt(0.5), rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, math.sqrt(0.5), rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_right_diagonal_movement_when_modules_are_pointing_angled_with_velocity():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(315),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, math.sqrt(0.5), rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, -math.sqrt(0.5), rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_show_pure_rotation_movement_when_modules_are_pointing_left_diagonal_with_velocity():
    drive_modules = create_drive_modules(1.0, 1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(45 + i * 90),
            0.0,
            0.0,
            0.0,
            math.sqrt(0.5),
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)

    assert math.isclose(motion.angular_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 1.0, rel_tol=1e-6, abs_tol=1e-15)

# state_of_wheel_modules_from_body_motion

def test_should_have_parallel_forward_wheels_with_forward_velocity_when_forward_motion():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, 0.0, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isclose(module_state[1].steering_angle_in_radians, math.pi, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_have_parallel_forward_wheels_with_legal_forward_velocity_when_excessive_forward_motion():
    drive_modules = create_drive_modules(drive_max_velocity=1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, 0.0, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isclose(module_state[1].steering_angle_in_radians, math.pi, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_have_parallel_forward_wheels_with_reverse_velocity_when_backward_motion():
    drive_modules = create_drive_modules(1.0, 1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, math.pi, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isclose(module_state[1].steering_angle_in_radians, 0.0, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_have_parallel_left_sideways_wheels_with_forward_velocity_when_sideways_motion():
    drive_modules = create_drive_modules(1.0, 1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, math.radians(90), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isclose(module_state[1].steering_angle_in_radians, math.radians(-90), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_have_parallel_right_sideways_wheels_with_forward_velocity_when_sideways_motion():
    drive_modules = create_drive_modules(1.0, 1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, math.radians(-90), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isclose(module_state[1].steering_angle_in_radians, math.radians(90), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_have_parallel_diagonal_wheels_with_forward_velocity_when_diagonal_motion():
    drive_modules = create_drive_modules(1.0, 1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, math.radians(45), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isclose(module_state[1].steering_angle_in_radians, math.radians(-135), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_have_angled_wheels_with_forward_velocity_when_pure_rotation():
    drive_modules = create_drive_modules(1.0, 1.0)
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        forward_angle = 45 + i * 90
        if forward_angle >= 180:
            forward_angle -= 360
        assert math.isclose(module_state[0].steering_angle_in_radians, math.radians(forward_angle), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, math.sqrt(0.5), rel_tol=1e-6, abs_tol=1e-15)

        reversing_angle = 45 + i * 90 + 180
        if reversing_angle >= 180:
            reversing_angle -= 360
        assert math.isclose(module_state[1].steering_angle_in_radians, math.radians(reversing_angle), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -math.sqrt(0.5), rel_tol=1e-6, abs_tol=1e-15)

def test_should_not_move_wheels_when_zero_motion():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isinf(module_state[0].steering_angle_in_radians)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 0.0, rel_tol=1e-6, abs_tol=1e-15)

        assert math.isinf(module_state[1].steering_angle_in_radians)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, 0.0, rel_tol=1e-6, abs_tol=1e-15)

def test_should_not_move_wheels_when_stopping():
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(45 + i * 90),
            0.0,
            0.0,
            0.0,
            math.sqrt(0.5),
            0.0,
            0.0,
        )
        states.append(module_state)

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == len(drive_modules)

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        assert math.isclose(module_state[0].steering_angle_in_radians, float('inf'), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 0.0, rel_tol=1e-6, abs_tol=1e-15)

        reversing_angle = 45 + i * 90 + 180
        if reversing_angle >= 360:
            reversing_angle -= 360
        assert math.isclose(module_state[1].steering_angle_in_radians, float('-inf'), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, 0.0, rel_tol=1e-6, abs_tol=1e-15)


# =============================================================================
# TWO-WHEEL TESTS
# =============================================================================

def test_should_reject_single_wheel_configuration():
    """Test that the controller rejects configurations with fewer than 2 wheels."""
    single_module = [create_drive_module("only", 0.0, 0.0)]

    with pytest.raises(ValueError) as excinfo:
        MultiWheelSteeringControlModel(single_module)

    assert "At least 2 drive modules are required" in str(excinfo.value)


def test_should_accept_two_wheel_configuration():
    """Test that the controller accepts a 2-wheel configuration."""
    drive_modules = create_two_wheel_drive_modules()
    controller = MultiWheelSteeringControlModel(drive_modules)

    assert controller.wheel_count == 2


def test_two_wheel_should_show_forward_movement_when_modules_pointing_forward():
    """Test forward kinematics for 2-wheel: forward motion."""
    drive_modules = create_two_wheel_drive_modules()
    controller = MultiWheelSteeringControlModel(drive_modules)

    states: List[DriveModuleMeasuredValues] = []
    for i in range(len(drive_modules)):
        module_state = DriveModuleMeasuredValues(
            drive_modules[i].name,
            drive_modules[i].steering_axis_xy_position.x,
            drive_modules[i].steering_axis_xy_position.y,
            math.radians(0),  # pointing forward
            0.0,
            0.0,
            0.0,
            1.0,  # moving forward
            0.0,
            0.0,
        )
        states.append(module_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    assert math.isclose(motion.linear_velocity.x, 1.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(motion.angular_velocity.z, 0.0, rel_tol=1e-6, abs_tol=1e-15)


def test_two_wheel_should_show_rotation_when_modules_pointing_perpendicular():
    """Test forward kinematics for 2-wheel: pure rotation (wheels pointing perpendicular to each other)."""
    drive_modules = create_two_wheel_drive_modules(length=1.0)
    controller = MultiWheelSteeringControlModel(drive_modules)

    # For pure rotation around center with front/rear wheels on centerline:
    # Front wheel at (0.5, 0) should point at 90 degrees (left/+y direction)
    # Rear wheel at (-0.5, 0) should point at -90 degrees (right/-y direction)
    # Both moving forward at velocity v creates rotation omega = v / r where r = 0.5
    states: List[DriveModuleMeasuredValues] = []

    # Front wheel points +y (90 degrees), moving forward
    front_state = DriveModuleMeasuredValues(
        drive_modules[0].name,
        drive_modules[0].steering_axis_xy_position.x,
        drive_modules[0].steering_axis_xy_position.y,
        math.radians(90),
        0.0, 0.0, 0.0,
        0.5,  # velocity
        0.0, 0.0,
    )
    states.append(front_state)

    # Rear wheel points -y (-90 degrees), moving forward
    rear_state = DriveModuleMeasuredValues(
        drive_modules[1].name,
        drive_modules[1].steering_axis_xy_position.x,
        drive_modules[1].steering_axis_xy_position.y,
        math.radians(-90),
        0.0, 0.0, 0.0,
        0.5,  # velocity
        0.0, 0.0,
    )
    states.append(rear_state)

    motion = controller.body_motion_from_wheel_module_states(states)

    # With wheels at +/-0.5 on x-axis, rotating at v=0.5, omega should be 1.0 rad/s
    assert math.isclose(motion.linear_velocity.x, 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(motion.linear_velocity.y, 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(motion.angular_velocity.z, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_two_wheel_should_have_parallel_forward_wheels_when_forward_motion():
    """Test inverse kinematics for 2-wheel: forward motion results in parallel forward-pointing wheels."""
    drive_modules = create_two_wheel_drive_modules()
    controller = MultiWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        1.0,  # v_x
        0.0,  # v_y
        0.0,  # omega
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == 2

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        # Forward state: angle = 0, velocity = 1.0
        assert math.isclose(module_state[0].steering_angle_in_radians, 0.0, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)

        # Reverse state: angle = pi, velocity = -1.0
        assert math.isclose(module_state[1].steering_angle_in_radians, math.pi, rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[1].drive_velocity_in_meters_per_second, -1.0, rel_tol=1e-6, abs_tol=1e-15)


def test_two_wheel_should_have_sideways_wheels_when_sideways_motion():
    """Test inverse kinematics for 2-wheel: sideways motion results in sideways-pointing wheels."""
    drive_modules = create_two_wheel_drive_modules()
    controller = MultiWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,  # v_x
        1.0,  # v_y (left)
        0.0,  # omega
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == 2

    for i in range(len(proposed_states)):
        module_state = proposed_states[i]

        # Forward state: angle = 90 degrees, velocity = 1.0
        assert math.isclose(module_state[0].steering_angle_in_radians, math.radians(90), rel_tol=1e-6, abs_tol=1e-15)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)


def test_two_wheel_should_have_angled_wheels_when_pure_rotation():
    """Test inverse kinematics for 2-wheel: pure rotation results in perpendicular wheel angles."""
    drive_modules = create_two_wheel_drive_modules(length=1.0)
    controller = MultiWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0,  # v_x
        0.0,  # v_y
        1.0,  # omega (counter-clockwise)
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == 2

    # Front wheel at (0.5, 0): v_x = 0 - 1.0*0 = 0, v_y = 0 + 1.0*0.5 = 0.5
    # Wheel should point at 90 degrees with velocity 0.5
    front_state = proposed_states[0]
    assert math.isclose(front_state[0].steering_angle_in_radians, math.radians(90), rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(front_state[0].drive_velocity_in_meters_per_second, 0.5, rel_tol=1e-6, abs_tol=1e-15)

    # Rear wheel at (-0.5, 0): v_x = 0 - 1.0*0 = 0, v_y = 0 + 1.0*(-0.5) = -0.5
    # Wheel should point at -90 degrees with velocity 0.5
    rear_state = proposed_states[1]
    assert math.isclose(rear_state[0].steering_angle_in_radians, math.radians(-90), rel_tol=1e-6, abs_tol=1e-15)
    assert math.isclose(rear_state[0].drive_velocity_in_meters_per_second, 0.5, rel_tol=1e-6, abs_tol=1e-15)


def test_two_wheel_not_move_wheels_when_zero_motion():
    """Test inverse kinematics for 2-wheel: zero motion results in infinity steering angles."""
    drive_modules = create_two_wheel_drive_modules()
    controller = MultiWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == 2

    for module_state in proposed_states:
        assert math.isinf(module_state[0].steering_angle_in_radians)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 0.0, rel_tol=1e-6, abs_tol=1e-15)


def test_two_wheel_velocity_normalization_when_excessive_motion():
    """Test that 2-wheel configuration normalizes velocities when they exceed motor limits."""
    drive_modules = create_two_wheel_drive_modules(drive_max_velocity=1.0)
    controller = MultiWheelSteeringControlModel(drive_modules)

    motion = BodyMotion(
        2.0,  # v_x (exceeds max)
        0.0,
        0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )

    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == 2

    for module_state in proposed_states:
        # Velocity should be normalized to max (1.0)
        assert math.isclose(module_state[0].drive_velocity_in_meters_per_second, 1.0, rel_tol=1e-6, abs_tol=1e-15)


def test_backward_compatible_alias():
    """Test that SimpleFourWheelSteeringControlModel still works as an alias."""
    drive_modules = create_drive_modules()
    controller = SimpleFourWheelSteeringControlModel(drive_modules)

    assert controller.wheel_count == 4

    # Should work identically to MultiWheelSteeringControlModel
    motion = BodyMotion(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    proposed_states = controller.state_of_wheel_modules_from_body_motion(motion)

    assert len(proposed_states) == 4
