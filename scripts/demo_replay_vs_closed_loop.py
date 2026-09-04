#!/usr/bin/env python3
"""Illustrate when generated interaction leaves logged human support."""
from strategic_agents.evaluation.replay import support_trace

logged = ["a1", "a2", "a3"]
teacher_forced = ["a1", "a2", "a3"]
closed_loop = ["a1", "b2", "b3"]
print("teacher-forced:", support_trace(logged, teacher_forced))
print("closed-loop:   ", support_trace(logged, closed_loop))
print("After b2 != a2, the log supplies no human response conditioned on b2.")
