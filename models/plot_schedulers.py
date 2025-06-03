from matplotlib import pyplot as plt

from models.guidance_schedulers import baseline_scheduler, linear_increasing_scheduler, cosine_increasing_scheduler

N_STEPS = 10

step_values = list(range(1, N_STEPS + 1))

plt.plot(step_values, [baseline_scheduler(x, N_STEPS, 7.5) for x in step_values], label='Baseline')
plt.plot(step_values, [linear_increasing_scheduler(x, N_STEPS, 7.5) for x in step_values], label='Linear')
plt.plot(step_values, [cosine_increasing_scheduler(x, N_STEPS, 7.5) for x in step_values], label='Cosine')
plt.ylabel('values')
plt.xlabel('steps')
plt.legend()
plt.show()
