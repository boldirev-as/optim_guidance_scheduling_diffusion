from prev_exp.models.guidance_schedulers import get_guidance_scheduler

total_steps = 50

guidance_scheduler_fn = get_guidance_scheduler('linear')
linear_values = [guidance_scheduler_fn(i, total_steps, 2) for i in range(total_steps)]
guidance_scheduler_fn = get_guidance_scheduler('cosine')
cosine_values = [guidance_scheduler_fn(i, total_steps, 2) for i in range(total_steps)]
guidance_scheduler_fn = get_guidance_scheduler('baseline')
baseline_values = [guidance_scheduler_fn(i, total_steps, 2) for i in range(total_steps)]

import matplotlib.pyplot as plt

plt.plot(linear_values, label='Linear')
plt.plot(cosine_values, label='Cosine')
plt.plot(baseline_values, label='Baseline')

plt.ylabel('Weights')
plt.xlabel('Steps')
plt.title('Guidance Schedulers')

plt.legend()
# plt.show()

plt.savefig('plots/schedulers.png', bbox_inches='tight', dpi=300)
