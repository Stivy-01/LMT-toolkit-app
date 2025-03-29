# -*- coding: utf-8 -*-
"""
Behavior analysis module for LMT data processing.
Contains functions for analyzing animal behavior patterns.
"""

from .behavior_processor import BehaviorProcessor
from .behavior_processor_hourly import BehaviorProcessor as BehaviorProcessorHourly
from .behavior_processor_interval import BehaviorProcessor as BehaviorProcessorInterval

__all__ = [
    'BehaviorProcessor',
    'BehaviorProcessorHourly',
    'BehaviorProcessorInterval'
]
