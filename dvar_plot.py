
import pandas as pd
import matplotlib.pyplot as plt

dvar_1 = pd.read_csv('Opt_dvars_risk_0.50.csv')
dvar_2 = pd.read_csv('Opt_dvars_risk_0.70.csv')
dvar_3 = pd.read_csv('Opt_dvars_risk_0.90.csv')

# sorted_1 = dvar_1.sort_values(by=['dvar0'], axis=0).loc[:, 'dvar0']
# sorted_2 = dvar_2.sort_values(by=['dvar0'], axis=0).loc[:, 'dvar0']
# sorted_3 = dvar_3.sort_values(by=['dvar0'], axis=0).loc[:, 'dvar0']
#
# plt.plot(sorted_1, sorted_1, 'o')
# plt.plot(sorted_2, sorted_2, 'o')
# plt.plot(sorted_3, sorted_3, 'o')
name = 'dvar0'

sorted_1 = dvar_1.sort_values(by=[name], axis=0).loc[:, name]
sorted_2 = dvar_2.sort_values(by=[name], axis=0).loc[:, name]
sorted_3 = dvar_3.sort_values(by=[name], axis=0).loc[:, name]

plt.plot(sorted_1, sorted_1, 'o')
plt.plot(sorted_2, sorted_2, 'o')
plt.plot(sorted_3, sorted_3, 'o')

plt.show()