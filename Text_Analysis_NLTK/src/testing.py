#%%
import src.common_function as cf 

a = "cat like to eat fish , fish is nice."

print('noun',cf.check_pos_tag(a, 'noun'))
print('verb',cf.check_pos_tag(a, 'verb'))
print('adj',cf.check_pos_tag(a, 'adj'))
print('adv',cf.check_pos_tag(a, 'adv'))
print('pron',cf.check_pos_tag(a, 'pron'))