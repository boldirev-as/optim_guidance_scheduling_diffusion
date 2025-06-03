from matplotlib import pyplot as plt

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

plt.plot(baseline_clip, baseline_fid, label='Baseline', marker='o')
plt.plot(linear_clip, linear_fid, label='Linear', marker='o')
plt.plot(cos_clip, cos_fid, label='Cosine', marker='o')
plt.xlim(0.25, 0.262)
plt.ylim(25, 37)
plt.ylabel('FID')
plt.xlabel('CLIP-Score')
plt.legend()
plt.show()
