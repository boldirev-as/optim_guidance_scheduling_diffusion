from matplotlib import pyplot as plt
from prev_exp.models.guidance_schedulers import get_guidance_scheduler

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

total_steps = 50

guidance_scheduler_fn = get_guidance_scheduler('linear')
linear_values = [guidance_scheduler_fn(i, total_steps, 2) for i in range(total_steps)]
guidance_scheduler_fn = get_guidance_scheduler('cosine')
cosine_values = [guidance_scheduler_fn(i, total_steps, 2) for i in range(total_steps)]
guidance_scheduler_fn = get_guidance_scheduler('baseline')
baseline_values = [guidance_scheduler_fn(i, total_steps, 2) for i in range(total_steps)]

ax2.plot(linear_values, label='Linear')
ax2.plot(cosine_values, label='Cosine')
ax2.plot(baseline_values, label='Baseline')

ax2.set_ylabel('Weights')
ax2.set_xlabel('Steps')
ax2.set_title('Guidance Schedulers')

ax2.legend()

baseline_fid = []
baseline_clip = []

linear_fid = []
linear_clip = []

cos_fid = []
cos_clip = []

with open('slum2852892.out', mode='r') as f:
    for line in f:
        if line.startswith('baseline') or line.startswith('linear') or line.startswith('cosine'):
            name, w, fid, _, _, clip, _, _ = line.split()
            if name == 'baseline':
                baseline_fid.append(float(fid))
                baseline_clip.append(float(clip))
            elif name == 'linear':
                linear_fid.append(float(fid))
                linear_clip.append(float(clip))
            elif name == 'cosine':
                cos_fid.append(float(fid))
                cos_clip.append(float(clip))

ax1.plot(baseline_clip, baseline_fid, label='Baseline', marker='o')
ax1.plot(linear_clip, linear_fid, label='Linear', marker='o')
ax1.plot(cos_clip, cos_fid, label='Cosine', marker='o')
ax1.set_xlim(0.25, 0.262)
ax1.set_ylim(25, 37)
ax1.set_ylabel('FID')
ax1.set_xlabel('CLIP-Score')
ax1.legend()
# plt.show()

ax1.set_title('FID vs. CLIP-Score')

plt.savefig('plots/scheduler_diff.png', bbox_inches='tight', dpi=300)
