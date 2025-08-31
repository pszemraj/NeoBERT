warmup_steps = 9
decay_steps = 198
constant_steps = 0

# This is the assertion from the scheduler
result = (constant_steps == 0 and warmup_steps < decay_steps) or (
    warmup_steps < constant_steps and constant_steps < decay_steps
)

print(f"warmup_steps: {warmup_steps}")
print(f"decay_steps: {decay_steps}")
print(f"constant_steps: {constant_steps}")
print(f"Assertion passes: {result}")
print(f"Condition 1: constant_steps == 0: {constant_steps == 0}")
print(f"Condition 2: warmup_steps < decay_steps: {warmup_steps < decay_steps}")
