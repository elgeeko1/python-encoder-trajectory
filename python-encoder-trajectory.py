#!/usr/bin/env python

import time
import datetime
import math
import numpy
from typing import Callable

class TimeStampedCircularBuffer:
    """
    A circular buffer that holds timestamped data entries.
    Once the buffer reaches its capacity, it starts overwriting the oldest data.
    """
    def __init__(self, capacity: int):
        """
        Initializes the buffer with a given capacity.
        
        :param capacity: The maximum number of timestamped data entries the buffer can hold.
        """
        self._capacity = capacity
        self._buffer = numpy.empty((0, 2), float)  # Initialize an empty 2D array for timestamp and data
        self._head = 0  # Points to the start of the buffer (write head)

    def append(self, timestamp: float, value: float) -> None:
        """
        Appends a new timestamped data entry to the buffer.

        :param timestamp: The timestamp of the data point.
        :param value: The data value to store.
        """
        if self._buffer.shape[0] < self._capacity:
            self._buffer = numpy.vstack((self._buffer, numpy.array([timestamp, value])))
        else:
            self._buffer[self._head] = [timestamp, value]
            self._head = (self._head + 1) % self._capacity

    def last(self) -> numpy.ndarray:
        """
        Returns the last appended data entry.

        :return: The last data entry in the buffer.
        """
        return self._buffer[(self._head - 1) % self._capacity]

    def values(self) -> numpy.ndarray:
        """
        Retrieves all the timestamped data entries in the buffer, ordered by the time they were added.
        This method takes into account the circular nature of the buffer to return the values in the correct order.

        :return: A numpy array of timestamped data entries, where each entry is a [timestamp, value] pair.
        """
        return numpy.roll(self._buffer, -self._head, axis=0)


class ExponentialSmoothing:
    """
    Exponential smoothing algorithm for time series data.
    """
    def __init__(self, alpha: float = 0.5, zero_floor = -math.inf):
        """
        Initializes the exponential smoothing filter.

        :param alpha: The smoothing factor, a value between 0 and 1.
        :param zero_floor: Threshold below which the smoothed value should saturate to zero
        """
        self._alpha = alpha
        self._last_smoothed = None
        self._zero_floor = zero_floor

    def smooth(self, value: float) -> float:
        """
        Applies exponential smoothing to the given value.

        :param value: The data value to be smoothed.
        :return: The smoothed data value.
        """
        if self._last_smoothed is None:
            self._last_smoothed = value
        else:
            self._last_smoothed = self._alpha * value + (1 - self._alpha) * self._last_smoothed
            if value < self._zero_floor:
                self._last_smoothed = 0.0

        return self._last_smoothed
    
    def value(self) -> float:
        """
        Returns the last smoothed value.

        :return: The last smoothed data value.
        """
        return self._last_smoothed

class Trajectory:
    """
    The Trajectory class calculates and stores the position and speed of an object over time.
    It uses a TimeStampedCircularBuffer to maintain a fixed-size window of the latest position and speed data points
    and applies exponential smoothing to the speed data to reduce noise and variability in the measurements.
    """

    def __init__(self, window_size: int, alpha: float = 0.5):
        """
        Initializes the Trajectory with a specified window size for the buffers and a smoothing factor for speed calculation.

        :param window_size: The maximum number of entries for the position and speed graphs.
        :param alpha: The smoothing factor used for exponential smoothing of the speed data.
        """
        self._last_timestamp = time.time()  # Stores the timestamp of the last update
        self._last_position = 0.0  # Stores the last calculated position
        self._last_speed = 0.0  # Stores the last calculated speed
        self._position_graph = TimeStampedCircularBuffer(window_size)  # Circular buffer for position data
        self._speed_graph = TimeStampedCircularBuffer(window_size)  # Circular buffer for speed data
        self._speed_filter = ExponentialSmoothing(alpha)  # Exponential smoothing filter for speed
        self._step_callbacks = []
        
    def register_callback(self, callback: Callable[[float, float, float], None]):
        """
        Register a callback to be issued upon every trajectory step.

        :param callback: method witih signature (float, float, float) -> None
        """
        self._step_callbacks.append(callback)
       
    def step(self, step_pulses: float = 1.0, offset_s: float = 0.0):
        """
        Updates the position and speed based on the step pulses received since the last update.

        :param step_pulses: The number of pulses since the last update, which is proportional to the distance moved.
        :param offset_s: The time offset in seconds to be added to the current time, for timestamping the update.
        """
        timestamp = time.time() + offset_s
        dt = timestamp - self._last_timestamp
        new_position = self._last_position + step_pulses
        new_speed = self._speed_filter.smooth(step_pulses / dt)
        self._position_graph.append(timestamp, new_position)
        self._speed_graph.append(timestamp, new_speed)
        self._last_timestamp = timestamp
        self._last_position = new_position
        self._last_speed = new_speed

        # issue callbacks
        for callback in self._step_callbacks:
            callback(timestamp, new_position, new_speed)

    def position(self) -> (float, float):
        """
        Returns the last recorded position and its associated timestamp.

        :return: A tuple containing the last timestamp and the last position.
        """
        return self._last_timestamp, self._last_position

    def positions(self) -> numpy.ndarray:
        """
        Retrieves all the recorded positions from the position graph.

        :return: A numpy array of timestamped position entries.
        """
        return self._position_graph.values()

    def speed(self) -> (float, float):
        """
        Returns the last recorded speed and its associated timestamp.

        :return: A tuple containing the last timestamp and the last speed.
        """
        return self._last_timestamp, self._last_speed

    def speeds(self) -> numpy.ndarray:
        """
        Retrieves all the recorded speeds from the speed graph.

        :return: A numpy array of timestamped speed entries.
        """
        return self._speed_graph.values()
    


def trajectory_callback(timestamp: float, position: float, speed: float) -> None:
    timestr = datetime.datetime.fromtimestamp(timestamp).strftime('%c')
    print(timestr + ": position " + str(position) + " with speed " + str(speed))

if __name__ == '__main__':
    trajectory = Trajectory(10, 0.5)
    trajectory.register_callback(trajectory_callback)
    for i in range(10):
        time.sleep(0.1)
        trajectory.step()
        
