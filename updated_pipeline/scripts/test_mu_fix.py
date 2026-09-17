import numpy as np, sys
sys.path.insert(0,'src')
from m2_label_generation.generative_model import _e_step, _m_step

# Simulate the fallback EM with the entity_tags has_organization matrix
# Just use the has_org columns from the .npy
et = np.load('outputs/m2_lf_matrices/entity_tags_lambda_matrix.npy')
M_org = et[:, 2:4]  # has_org_pos, has_org_neg columns

print(f'Matrix shape: {M_org.shape}')
print(f'Positive votes (col 0 == 1): {np.sum(M_org[:,0]==1)}')
print(f'Negative votes (col 1 == 0): {np.sum(M_org[:,1]==0)}')
print(f'Abstain (col 0 == -1): {np.sum(M_org[:,0]==-1)}')
print()

# Current _fit_fallback behavior
from m2_label_generation.generative_model import _safeguard_probs, ABSTAIN

n, m = M_org.shape
num_classes = 2
n_iterations = 8

# Current behavior: mu is updated
mu = np.full(num_classes, 1.0/num_classes)
theta = np.full(m, 0.7)
probs = np.full((n, num_classes), 1.0/num_classes)
for _ in range(max(1, n_iterations)):
    probs = _e_step(M_org, mu, theta, num_classes)
    mu, theta = _m_step(M_org, probs, num_classes)

print(f'CURRENT (mu updated): method=fallback_generative')
print(f'  class_priors: {mu}')
print(f'  row 81844 probs: {probs[81844]}')
print(f'  row 81844 has_organization (col 1): {probs[81844, 1]}')
print()

# Fixed behavior: mu stays uniform
mu_fixed = np.full(num_classes, 1.0/num_classes)
theta_fixed = np.full(m, 0.7)
probs_fixed = np.full((n, num_classes), 1.0/num_classes)
for _ in range(max(1, n_iterations)):
    probs_fixed = _e_step(M_org, mu_fixed, theta_fixed, num_classes)
    # DON'T update mu - keep it uniform
    theta_fixed = _m_step(M_org, probs_fixed, num_classes)[1]  # only update theta

print(f'FIXED (mu uniform): method=fallback_generative')
print(f'  class_priors: {mu_fixed}')
print(f'  row 81844 probs: {probs_fixed[81844]}')
print(f'  row 81844 has_organization (col 1): {probs_fixed[81844, 1]}')
print()

# Check: how many records now have has_organization > 0.5?
print(f'CURRENT: {np.mean(probs[:,1] > 0.5)*100:.1f}% have has_organization > 0.5')
print(f'FIXED: {np.mean(probs_fixed[:,1] > 0.5)*100:.1f}% have has_organization > 0.5')

# Check a record with negative vote
neg_rows = np.where((M_org[:,0]==-1) & (M_org[:,1]==0))[0]
if len(neg_rows) > 0:
    r = neg_rows[0]
    print(f'\nRecord with negative vote (row {r}):')
    print(f'  CURRENT probs: {probs[r]}')
    print(f'  FIXED probs: {probs_fixed[r]}')

# Check a record with both abstain
abstain_rows = np.where((M_org[:,0]==-1) & (M_org[:,1]==-1))[0]
if len(abstain_rows) > 0:
    r = abstain_rows[0]
    print(f'\nRecord with both abstain (row {r}):')
    print(f'  CURRENT probs: {probs[r]}')
    print(f'  FIXED probs: {probs_fixed[r]}')
